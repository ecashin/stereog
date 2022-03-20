use lv2::prelude::*;
use std::collections::HashMap;
use wmidi::*;

const MAX_SAMPLE_SECONDS: f32 = 5.0;

#[derive(PortCollection)]
pub struct Ports {
    control: InputPort<AtomPort>,
    in_left: InputPort<Audio>,
    in_right: InputPort<Audio>,
    out_left: OutputPort<Audio>,
    out_right: OutputPort<Audio>,
}

#[derive(FeatureCollection)]
pub struct Features<'a> {
    map: LV2Map<'a>,
}

#[derive(URIDCollection)]
pub struct URIDs {
    atom: AtomURIDCollection,
    midi: MidiURIDCollection,
    unit: UnitURIDCollection,
}

enum SamplerState {
    Off,
    Armed,
    Recording,
    Active,
}

#[uri("https://github.com/ecashin/lrgran")]
pub struct Lrgran {
    sample_rate: usize,
    active_notes: HashMap<wmidi::Note, wmidi::Velocity>,
    sample: Option<(Vec<f32>, Vec<f32>)>,
    sampler_state: SamplerState,
    sound_onset: usize,
    record_pos: usize,
    urids: URIDs,
}

impl Lrgran {
    fn arm_sampler(&mut self) {
        let n_frames = (self.sample_rate as f32 * MAX_SAMPLE_SECONDS) as usize;
        let left: Vec<f32> = vec![0.0; n_frames];
        let right: Vec<f32> = vec![0.0; n_frames];
        self.sample = Some((left, right));
        self.sampler_state = SamplerState::Armed;
    }

    // A function to write a chunk of output, to be called from `run()`. If the gate is high, then the input will be passed through for this chunk, otherwise silence is written.
    fn write_output(&mut self, ports: &mut Ports, offset: usize, mut len: usize) {
        if ports.in_left.len() < offset + len {
            len = ports.in_left.len() - offset;
        }

        let active = !self.active_notes.is_empty();

        let in_left = &ports.in_left[offset..offset + len];
        let in_right = &ports.in_right[offset..offset + len];
        let out_left = &mut ports.out_left[offset..offset + len];
        let out_right = &mut ports.out_right[offset..offset + len];

        if let Some(sample) = &mut self.sample {
            match self.sampler_state {
                SamplerState::Armed | SamplerState::Recording => {
                    let left = &mut sample.0;
                    let right = &mut sample.1;
                    for (sample_left, sample_right) in in_left.iter().zip(in_right.iter()) {
                        left[self.record_pos] = *sample_left;
                        right[self.record_pos] = *sample_right;
                        self.record_pos = (self.record_pos + 1) % left.len();
                    }
                }
                _ => (),
            }
        }
        if active {
            out_left.copy_from_slice(in_left);
            out_right.copy_from_slice(in_right);
        } else {
            for (frame_left, frame_right) in out_left.iter_mut().zip(out_right.iter_mut()) {
                *frame_left = 0.0;
                *frame_right = 0.0;
            }
        }
    }
}

impl Plugin for Lrgran {
    type Ports = Ports;

    type InitFeatures = Features<'static>;
    type AudioFeatures = ();

    fn new(plugin_info: &PluginInfo, features: &mut Features<'static>) -> Option<Self> {
        println!("lrgran new");
        Some(Self {
            sample_rate: plugin_info.sample_rate() as usize,
            active_notes: HashMap::new(),
            sample: None,
            sampler_state: SamplerState::Off,
            record_pos: 0,
            sound_onset: 0,
            urids: features.map.populate_collection()?,
        })
    }

    fn run(&mut self, ports: &mut Ports, _: &mut (), _: u32) {
        let mut offset: usize = 0;

        let control_sequence = ports
            .control
            .read(self.urids.atom.sequence, self.urids.unit.beat)
            .unwrap();

        for (timestamp, message) in control_sequence {
            let timestamp: usize = if let Some(timestamp) = timestamp.as_frames() {
                timestamp as usize
            } else {
                continue;
            };

            let message = if let Some(message) = message.read(self.urids.midi.wmidi, ()) {
                message
            } else {
                continue;
            };

            match message {
                MidiMessage::NoteOn(ch, note, vel) => {
                    println!("ON ch:{:?} note:{:?} vel:{:?}", ch, note, vel);
                    println!("notes:{:?}", self.active_notes);
                    self.active_notes.insert(note, vel);
                    if note == Note::A4 {
                        self.arm_sampler();
                    }
                }
                MidiMessage::NoteOff(ch, note, vel) => {
                    println!("OFF ch:{:?} note:{:?} vel:{:?}", ch, note, vel);
                    self.active_notes.remove(&note);
                }
                MidiMessage::ProgramChange(ch, program) => {
                    println!("PC ch:{:?} program:{:?}", ch, program);
                    self.arm_sampler();
                }
                _ => (),
            }
            self.write_output(ports, offset, timestamp + offset);
            offset = timestamp;
        }
        assert_eq!(ports.in_left.len(), ports.in_right.len());
        self.write_output(ports, offset, ports.in_left.len() - offset);
    }

    // During it's runtime, the host might decide to deactivate the plugin. When the plugin is reactivated, the host calls this method which gives the plugin an opportunity to reset it's internal state.
    fn activate(&mut self, _features: &mut Features<'static>) {
        println!("lrgran activate");
        self.active_notes = HashMap::new();
    }
}

lv2_descriptors!(Lrgran);

use lv2::prelude::*;
use std::collections::HashMap;
use wmidi::*;

#[derive(PortCollection)]
pub struct Ports {
    control: InputPort<AtomPort>,
    input: InputPort<Audio>,
    output: OutputPort<Audio>,
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

#[uri("https://github.com/ecashin/lrgran")]
pub struct Lrgran {
    active_notes: HashMap<wmidi::Note, wmidi::Velocity>,
    urids: URIDs,
}

impl Lrgran {
    // A function to write a chunk of output, to be called from `run()`. If the gate is high, then the input will be passed through for this chunk, otherwise silence is written.
    fn write_output(&mut self, ports: &mut Ports, offset: usize, mut len: usize) {
        if ports.input.len() < offset + len {
            len = ports.input.len() - offset;
        }

        let active = !self.active_notes.is_empty();

        let input = &ports.input[offset..offset + len];
        let output = &mut ports.output[offset..offset + len];

        if active {
            output.copy_from_slice(input);
        } else {
            for frame in output.iter_mut() {
                *frame = 0.0;
            }
        }
    }
}

impl Plugin for Lrgran {
    type Ports = Ports;

    type InitFeatures = Features<'static>;
    type AudioFeatures = ();

    fn new(_plugin_info: &PluginInfo, features: &mut Features<'static>) -> Option<Self> {
        println!("lrgran new");
        Some(Self {
            active_notes: HashMap::new(),
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
                }
                MidiMessage::NoteOff(ch, note, vel) => {
                    println!("OFF ch:{:?} note:{:?} vel:{:?}", ch, note, vel);
                    self.active_notes.remove(&note);
                }
                MidiMessage::ProgramChange(ch, program) => {
                    println!("PC ch:{:?} program:{:?}", ch, program);
                }
                _ => (),
            }

            self.write_output(ports, offset, timestamp + offset);
            offset = timestamp;
        }

        self.write_output(ports, offset, ports.input.len() - offset);
    }

    // During it's runtime, the host might decide to deactivate the plugin. When the plugin is reactivated, the host calls this method which gives the plugin an opportunity to reset it's internal state.
    fn activate(&mut self, _features: &mut Features<'static>) {
        println!("lrgran activate");
        self.active_notes = HashMap::new();
    }
}

lv2_descriptors!(Lrgran);

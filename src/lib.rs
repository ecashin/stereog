use lv2::prelude::*;
use rand::prelude::*;
use rand::Rng;
use std::{collections::HashMap, fmt};
use wmidi::*;

mod movavg;

use crate::movavg::MovAvgAbs;

const GRAINS_PER_SECOND: usize = 10;
const LOW_PITCH_HZ: usize = 100;
const MAX_SAMPLE_SECONDS: f32 = 5.0;
const N_GRAINS: usize = 5;
const UNHEARD_VALUE: f32 = -2.0; // for unused audio buffer slots

#[derive(PortCollection)]
pub struct Ports {
    control: InputPort<AtomPort>,
    threshold: InputPort<Control>,
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

#[derive(Debug, PartialEq)]
enum SamplerState {
    Armed,
    Recording,
    Playing,
}

fn play_speed_for_note(note: &Note) -> f32 {
    let ratio = note.to_freq_f64() / Note::C3.to_freq_f64();
    ratio as f32
}

fn interpolate(
    sound_start: usize,
    sound_end: usize,
    sound_pos: f32, // 1.0 ~ the whole sound
    ring_left: &[f32],
    ring_right: &[f32],
) -> (f32, f32) {
    let sound_len = (sound_end - sound_start) as f32;
    let pos = sound_pos * sound_len;
    let before = pos.floor();
    let after = before + 1.0;
    let before_wt = after - pos;
    let after_wt = pos - before;
    let before = before.round() as usize % ring_left.len();
    let after = after.round() as usize % ring_left.len();
    let left = ring_left[before] * before_wt + ring_left[after] * after_wt;
    let right = ring_right[before] * before_wt + ring_right[after] * after_wt;
    (left, right)
}

const TUKEY_WINDOW_ALPHA: f32 = 0.5;

fn tukey_window(pos: f32, len: f32) -> f32 {
    let transition = (len * TUKEY_WINDOW_ALPHA) / 2.0;
    let n = pos + 1.0; // using variable names from Wikipedia
    if n > transition && n < len - transition {
        1.0
    } else {
        let n = if n >= len - transition { len - n } else { n };
        (1.0 - ((2.0 * std::f32::consts::PI * n) / (TUKEY_WINDOW_ALPHA * len)).cos()) / 2.0
    }
}

struct Sampler {
    state: SamplerState,
    left: Vec<f32>,
    right: Vec<f32>,
    sample_rate: usize,
    record_pos: usize,
    recording_ma: MovAvgAbs,
    last_recording_ma: f32,
    sound_start: Option<usize>,
    sound_end: Option<usize>,
    granular: Option<Granular>,
}

impl fmt::Debug for Sampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sampler")
            .field("state", &self.state)
            .field("left", &self.left)
            .field("right", &self.right)
            .field("sample_rate", &self.sample_rate)
            .field("record_pos", &self.record_pos)
            .field("last_recording_ma", &self.last_recording_ma)
            .field("sound_start", &self.sound_start)
            .field("sound_end", &self.sound_end)
            .field("granular", &self.granular)
            .finish()
    }
}

// grains are sound-relative, abstracted from buffer wrapping
#[derive(Debug)]
struct Grain {
    start: usize, // the offset of the grain inside the sound
    end: usize,   // the end of the grain inside the sound
    pos: f32,     // the fractional playback position inside the sound
}

impl Grain {
    fn new(grain_len: usize, sound_len: usize) -> Self {
        assert!(grain_len > 1);
        assert!(grain_len < sound_len);
        let mut rng = thread_rng();
        let shrink = if grain_len > 20 {
            rng.gen_range(0..grain_len / 10)
        } else {
            0
        };
        let len = grain_len - shrink;
        let start = rng.gen_range(0..sound_len - len);
        Self {
            start,
            end: start + len,
            pos: 0.0,
        }
    }

    // returns sound offset and amplitude or None if grain is done
    fn next(&mut self, speed: f32) -> Option<(f32, f32)> {
        if self.pos >= 1.0 {
            None
        } else {
            let n_steps_unit_speed = (self.end - self.start) as f32;
            let n_steps = n_steps_unit_speed / speed;
            let pos = self.pos;
            self.pos += 1.0 / n_steps;
            let amp = tukey_window(pos * n_steps as f32, n_steps as f32);
            Some((pos, amp))
        }
    }
}

#[derive(Debug)]
struct Granular {
    grain_len: usize,
    sound_len: usize,
    grains: HashMap<wmidi::Note, Vec<Grain>>,
    left_mixer: Vec<f32>,
    right_mixer: Vec<f32>,
}

impl Granular {
    fn new(grain_len: usize, sound_len: usize, n_grains: usize) -> Self {
        Self {
            grain_len,
            sound_len,
            grains: HashMap::new(),
            left_mixer: vec![0.0; n_grains],
            right_mixer: vec![0.0; n_grains],
        }
    }

    fn next(
        &mut self,
        ring_left: &[f32],
        ring_right: &[f32],
        sound_start: usize,
        sound_end: usize,
        active_notes: &HashMap<wmidi::Note, wmidi::Velocity>,
    ) -> (f32, f32) {
        self.left_mixer.clear();
        self.right_mixer.clear();
        for (note, _) in active_notes.iter() {
            let speed = play_speed_for_note(note);
            let grains = self.grains.entry(*note).or_insert_with(|| {
                (0..N_GRAINS)
                    .map(|_| Grain::new(self.grain_len, self.sound_len))
                    .collect()
            });
            for grain in grains.iter_mut() {
                let mut g_next = grain.next(speed);
                if g_next.is_none() {
                    *grain = Grain::new(self.grain_len, self.sound_len);
                    g_next = grain.next(speed);
                }
                let (sound_pos, amplitude) = g_next.unwrap();
                let (lt, rt) =
                    interpolate(sound_start, sound_end, sound_pos, ring_left, ring_right);
                self.left_mixer.push(lt * amplitude);
                self.right_mixer.push(rt * amplitude);
            }
        }
        (
            self.left_mixer.iter().sum::<f32>() / (self.left_mixer.len() as f32),
            self.right_mixer.iter().sum::<f32>() / (self.right_mixer.len() as f32),
        )
    }
}

fn frames_to_seconds(sample_rate: usize, n_frames: usize) -> f32 {
    let sr = sample_rate as f32;
    let n = n_frames as f32;
    n / sr
}

impl Sampler {
    fn new(sample_rate: usize, sample_seconds: f32) -> Self {
        let n_frames = (sample_seconds * sample_rate as f32) as usize;
        println!(
            "creating sampler with {} stereo frames, {} seconds",
            n_frames,
            frames_to_seconds(sample_rate, n_frames)
        );
        let left = vec![UNHEARD_VALUE; n_frames];
        let right = vec![UNHEARD_VALUE; n_frames];
        let (_, recording_ma) = make_moving_average(sample_rate, LOW_PITCH_HZ, 0.0);
        Self {
            state: SamplerState::Armed,
            left,
            right,
            sample_rate,
            record_pos: 0,
            recording_ma,
            last_recording_ma: 0.0,
            sound_start: None,
            sound_end: None,
            granular: None,
        }
    }

    fn frame(&mut self, active_notes: &HashMap<wmidi::Note, wmidi::Velocity>) -> (f32, f32) {
        self.granular.as_mut().unwrap().next(
            &self.left,
            &self.right,
            self.sound_start.unwrap(),
            self.sound_end.unwrap(),
            active_notes,
        )
    }

    fn find_sound_start(&self, last_avg: f32, sound_end_threshold: f32) -> usize {
        let (_ma_len, mut ma) = make_moving_average(self.sample_rate, LOW_PITCH_HZ, last_avg);
        let mut most_quiet_pos: Option<usize> = None;
        let mut most_quiet_amp: Option<f32> = None;
        for i in 1..self.left.len() - 1 {
            let pos = (self.record_pos + self.left.len() - i) % self.left.len();
            if self.left[pos] == UNHEARD_VALUE {
                let one_right = (pos + 1) % self.left.len();
                println!(
                    "looking back, found unused audio buffer space at pos:{}; returning {}",
                    pos, one_right,
                );
                return one_right;
            }
            let mono = (self.left[pos] + self.right[pos]) / 2.0;
            let avg = ma.update(mono);
            if most_quiet_amp.is_none() || most_quiet_amp.unwrap() > mono {
                most_quiet_amp = Some(avg);
                most_quiet_pos = Some(pos);
            }
            if avg < sound_end_threshold {
                println!(
                    "early found sound start:{} in {} frames",
                    pos,
                    self.left.len()
                );
                return pos;
            }
        }
        if let Some(pos) = most_quiet_pos {
            println!(
                "late found sound start:{} in {} frames",
                pos,
                self.left.len()
            );
            pos
        } else {
            let pos = (self.record_pos + self.left.len() - 1) % self.left.len();
            println!("no quiet sound start found; using {}", pos);
            pos
        }
    }

    fn listen(
        &mut self,
        in_left: std::slice::Iter<'_, f32>,
        in_right: std::slice::Iter<'_, f32>,
        threshold: f32,
    ) {
        for (sample_left, sample_right) in in_left.zip(in_right) {
            self.left[self.record_pos] = *sample_left;
            self.right[self.record_pos] = *sample_right;
            let avg = self
                .recording_ma
                .update((*sample_left + *sample_right) / 2.0);
            match self.state {
                SamplerState::Armed => {
                    if avg > threshold && self.last_recording_ma < threshold {
                        self.sound_start =
                            Some(self.find_sound_start(avg, sound_end_threshold(threshold)));
                        self.state = SamplerState::Recording;
                        println!("changing from armed to recording sampler");
                    }
                }
                SamplerState::Recording => {
                    if avg < sound_end_threshold(threshold) {
                        let sound_end = self.record_pos;
                        let sound_start = self.sound_start.unwrap();
                        self.sound_end = Some(sound_end);
                        self.state = SamplerState::Playing;
                        println!(
                            "changing from recording to playing sampler with {}-second sound",
                            frames_to_seconds(self.sample_rate, sound_end - sound_start)
                        );
                        let end = if sound_end < sound_start {
                            sound_end + self.left.len()
                        } else {
                            sound_end
                        };
                        let sound_len = end - sound_start;
                        let granular =
                            Granular::new(self.grain_len(sound_len), sound_len, N_GRAINS);
                        self.granular = Some(granular);
                    }
                }
                SamplerState::Playing => (),
            }
            self.last_recording_ma = avg;
            self.record_pos = (self.record_pos + 1) % self.left.len();
        }
    }

    fn grain_len(&self, sound_len: usize) -> usize {
        let len = self.sample_rate / GRAINS_PER_SECOND;
        if sound_len / len > 3 {
            len
        } else {
            sound_len / 3
        }
    }
}

#[uri("https://github.com/ecashin/stereog")]
pub struct Stereog {
    sample_rate: usize,
    active_notes: HashMap<wmidi::Note, wmidi::Velocity>,
    sampler: Sampler,
    urids: URIDs,
}

fn make_moving_average(
    sample_rate: usize,
    example_hz: usize,
    initial_avg: f32,
) -> (usize, MovAvgAbs) {
    let len = sample_rate / (2 * example_hz);
    assert!(len >= 2);
    (len, MovAvgAbs::new(len, initial_avg))
}

impl Stereog {
    // A function to write a chunk of output, to be called from `run()`. If the gate is high, then the input will be passed through for this chunk, otherwise silence is written.
    fn write_output(&mut self, ports: &mut Ports, offset: usize, mut len: usize) {
        if ports.in_left.len() < offset + len {
            len = ports.in_left.len() - offset;
        }

        let notes_playing = !self.active_notes.is_empty();

        let in_left = &ports.in_left[offset..offset + len];
        let in_right = &ports.in_right[offset..offset + len];
        let out_left = &mut ports.out_left[offset..offset + len];
        let out_right = &mut ports.out_right[offset..offset + len];

        if self.sampler.state != SamplerState::Playing {
            self.sampler
                .listen(in_left.iter(), in_right.iter(), *ports.threshold);
        }
        if notes_playing && self.sampler.state == SamplerState::Playing {
            for (out_l, out_r) in out_left.iter_mut().zip(out_right.iter_mut()) {
                let (g_left, g_right) = self.sampler.frame(&self.active_notes);
                *out_l = g_left;
                *out_r = g_right;
            }
        } else {
            for (out_sample_left, out_sample_right) in out_left.iter_mut().zip(out_right.iter_mut())
            {
                *out_sample_left = 0.0;
                *out_sample_right = 0.0;
            }
        }
    }
}

fn sound_end_threshold(start_threshold: f32) -> f32 {
    start_threshold / 5.0
}

impl Plugin for Stereog {
    type Ports = Ports;

    type InitFeatures = Features<'static>;
    type AudioFeatures = ();

    fn new(plugin_info: &PluginInfo, features: &mut Features<'static>) -> Option<Self> {
        println!("stereog new");
        let sample_rate = plugin_info.sample_rate() as usize;
        Some(Self {
            sample_rate,
            active_notes: HashMap::new(),
            sampler: Sampler::new(sample_rate, MAX_SAMPLE_SECONDS),
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
                    self.active_notes.insert(note, vel);
                    println!("notes:{:?}", self.active_notes);
                    println!(
                        "sampler moving average: {}",
                        self.sampler.recording_ma.read()
                    );
                    if note == Note::A4 {
                        self.sampler = Sampler::new(self.sample_rate, MAX_SAMPLE_SECONDS);
                    }
                }
                MidiMessage::NoteOff(ch, note, vel) => {
                    println!("OFF ch:{:?} note:{:?} vel:{:?}", ch, note, vel);
                    self.active_notes.remove(&note);
                }
                MidiMessage::ProgramChange(ch, program) => {
                    println!("PC ch:{:?} program:{:?}", ch, program);
                    self.sampler = Sampler::new(self.sample_rate, MAX_SAMPLE_SECONDS);
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
        println!("stereog activate");
        self.active_notes = HashMap::new();
    }
}

lv2_descriptors!(Stereog);

#[cfg(test)]
mod test {
    use super::tukey_window;
    use super::{interpolate, Grain, Granular, Sampler, SamplerState};
    use std::collections::HashMap;
    use std::iter::Iterator;

    const TESTING_THRESHOLD: f32 = 0.4;

    #[test]
    fn test_interpolate() {
        let left: Vec<_> = (0..15).map(|i| i as f32).collect();
        let right = left.clone();
        let (lt, rt) = interpolate(5, 20, 0.3, &left[..], &right[..]);
        assert!((lt - rt).abs() < f32::EPSILON);
        assert!((lt - 0.299999).abs() < 0.00001);
    }

    #[test]
    fn test_grain() {
        let grain_len = 22050;
        let sound_len = 88200;
        let mut grain = Grain::new(grain_len, sound_len);
        println!("{:?}", grain);
        let g_next = grain.next(1.0);
        assert!(g_next.is_some());
        let (pos, amp) = g_next.unwrap();
        println!("pos:{} amp:{}", pos, amp);
        assert!(pos >= 0.0);
        assert!(pos < 1.0);
        for _ in 0..((grain.end - grain.start) / 2) {
            grain.next(1.0);
        }
        let (pos2, amp2) = grain.next(1.0).unwrap();
        println!("pos1:{} pos2:{} amp1:{} amp2:{}", pos, pos2, amp, amp2);
        assert!(pos < pos2);
        assert!(amp2 > amp);
    }

    #[test]
    fn test_varispeed_grain() {
        let grain_len = 22050;
        let sound_len = 88200;
        let mut grain = Grain::new(grain_len, sound_len);
        let g_next = grain.next(1.0);
        assert!(g_next.is_some());
        let (pos, _amp) = g_next.unwrap();
        assert!(pos >= 0.0);
        assert!(pos < 1.0);

        // double speed grain consumed in half the number of steps
        let speed = 2.0;
        for _ in 0..((grain.end - grain.start) / 2) {
            grain.next(speed);
        }
        assert!(grain.next(speed).is_none());

        // half speed grain takes twice as long to consume
        grain = Grain::new(grain_len, sound_len);
        let speed = 0.5;
        let n_steps = (grain.end - grain.start) as f32;
        for _ in 0..((n_steps * 1.8) as usize) {
            grain.next(speed);
        }
        assert!(grain.next(speed).is_some());
    }

    #[test]
    fn print_short_grain() {
        let show = move |mut g: Grain, i| loop {
            if let Some((pos, amp)) = g.next(1.0) {
                println!("grain:{} pos:{} amp:{}", i, pos, amp);
            } else {
                break;
            }
        };
        show(Grain::new(10, 100), 1);
        show(Grain::new(5, 100), 2);
    }

    #[test]
    fn test_tukey_window() {
        let start = tukey_window(0.0, 1000.0);
        let mid = tukey_window(499.0, 1000.0);
        let left = tukey_window(199.0, 1000.0);
        let right = tukey_window(799.0, 1000.0);
        let end = tukey_window(999.0, 1000.0);
        println!(
            "tukey for 1000: 0:{} 199:{} 499:{} 799:{} 999:{}",
            start, left, mid, right, end
        );
        assert!(start < left);
        assert!(start < mid);
        assert!(end < right);
        assert!(end < mid);
    }

    fn active_notes() -> HashMap<wmidi::Note, wmidi::Velocity> {
        HashMap::from([(wmidi::Note::B3, wmidi::Velocity::MAX)])
    }

    #[test]
    fn test_granular_zero() {
        let mut granular = Granular::new(30, 200, 3);
        let left = [0.0; 200];
        let right = [0.0; 200];
        for i in 1..=10 {
            let (lt, rt) = granular.next(&left, &right, 0, 100, &active_notes());
            println!("{} {} {}", i, lt, rt);
        }
    }

    #[test]
    fn test_granular_wrap() {
        let mut granular = Granular::new(30, 200, 3);
        let left = [0.0; 200];
        let right = [0.0; 200];
        for i in 1..=10 {
            let (lt, rt) = granular.next(&left, &right, 190, 210, &active_notes());
            println!("{} {} {}", i, lt, rt);
        }
    }

    struct WaveForm {
        period: usize,
        pos: usize,
    }

    impl WaveForm {
        fn new(period: usize) -> Self {
            Self { period, pos: 0 }
        }
    }

    impl<'a> Iterator for &'a mut WaveForm {
        type Item = f32;

        fn next(&mut self) -> Option<Self::Item> {
            let pos = self.pos as f32 / self.period as f32;
            self.pos += 1;
            let value = (2.0 * std::f32::consts::PI * pos).sin();
            Some(value)
        }
    }

    #[test]
    fn test_sampler_start_recording() {
        let sr = 2000;
        let mut sampler = Sampler::new(sr, 3.0);
        assert_eq!(sampler.state, SamplerState::Armed);
        let mut wav_left = WaveForm::new(20);
        let mut wav_right = WaveForm::new(20);
        let left: Vec<_> = wav_left.take(50).collect();
        let right: Vec<_> = wav_right.take(50).collect();
        sampler.listen(left[..].iter(), right[..].iter(), TESTING_THRESHOLD);
        println!("sampler:{:?}", sampler);
        assert_eq!(sampler.state, SamplerState::Recording);
        assert_eq!(sampler.sound_start, Some(0));
    }

    #[test]
    fn test_sampler_one_listen() {
        let sr = 2000;
        let mut sampler = Sampler::new(sr, 3.0);
        assert_eq!(sampler.state, SamplerState::Armed);
        let mut wav_left = WaveForm::new(20);
        let mut wav_right = WaveForm::new(20);
        let mut left: Vec<_> = wav_left.take(50).collect();
        let mut right: Vec<_> = wav_right.take(50).collect();
        sampler.listen(left[..].iter(), right[..].iter(), TESTING_THRESHOLD);
        println!("sampler:{:?}", sampler);
        left = wav_left.take(200).collect();
        right = wav_right.take(200).collect();
        let envelope: Vec<_> = (0..left.len())
            .map(|i| tukey_window(i as f32, left.len() as f32))
            .collect();
        for i in 0..100 {
            left[i] *= envelope[left.len() - 1 - (100 - i)];
            right[i] *= envelope[left.len() - 1 - (100 - i)];
        }
        sampler.listen(left[..].iter(), right[..].iter(), TESTING_THRESHOLD);
        assert_eq!(sampler.state, SamplerState::Playing);
    }
}

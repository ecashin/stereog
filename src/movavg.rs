pub struct MovAvg {
    len: f32,
    avg: f32,
}

impl MovAvg {
    pub fn new(window_size: usize, initial_avg: f32) -> Self {
        Self {
            len: window_size as f32,
            avg: initial_avg,
        }
    }

    pub fn update(&mut self, value: f32) -> f32 {
        let prev = self.avg * (self.len - 1.0) / self.len;
        let new = value / self.len;
        self.avg = prev + new;
        self.avg
    }

    pub fn read(&self) -> f32 {
        self.avg
    }
}

pub struct MovAvgAbs {
    ma: MovAvg,
}

impl MovAvgAbs {
    pub fn new(window_size: usize, initial_avg: f32) -> Self {
        Self {
            ma: MovAvg::new(window_size, initial_avg),
        }
    }

    pub fn update(&mut self, value: f32) -> f32 {
        self.ma.update(value.abs())
    }

    pub fn read(&self) -> f32 {
        self.ma.read()
    }
}

pub struct MovAvgExact {
    buf: Vec<f32>,
    sum: f32,
    pos: usize,
}

#[allow(dead_code)]
impl MovAvgExact {
    pub fn new(window_size: usize, initial_avg: f32) -> Self {
        let buf = vec![initial_avg; window_size];
        let sum = buf.iter().sum();
        Self { buf, sum, pos: 0 }
    }

    pub fn update(&mut self, value: f32) -> f32 {
        let oldest_pos = (self.pos + 1 + self.buf.len()) % self.buf.len();
        self.buf[self.pos] = value;
        self.sum = self.sum - self.buf[oldest_pos] + value;
        self.pos = (self.pos + 1) % self.buf.len();
        self.sum / self.buf.len() as f32
    }
}

#[cfg(test)]
mod test {
    use super::{MovAvg, MovAvgExact};

    #[test]
    fn test_movavg() {
        let mut ma = MovAvg::new(4, 10.0);
        assert_eq!(ma.avg, 10.0);
        assert_eq!(ma.update(50.0), 20.0);
    }

    #[test]
    fn test_movavg_env() {
        let len = 100;
        let mut amp = vec![1.0; len];
        for i in 0..20 {
            amp[i] = i as f32 / 20.0;
            amp[len - 1 - i] = amp[i];
        }
        let wavlen = 10.0;
        let wav: Vec<_> = (0..amp.len())
            .map(|i| {
                let sin = (((i as f32) * 2.0 * std::f32::consts::PI) / wavlen).sin();
                sin * amp[i]
            })
            .collect();
        let mut ma1 = MovAvg::new(20, 0.0);
        let mut ma2 = MovAvgExact::new(20, 0.0);
        let avgs1: Vec<_> = wav.iter().map(|i| ma1.update(i.abs())).collect();
        let avgs2: Vec<_> = wav.iter().map(|i| ma2.update(i.abs())).collect();
        let diffs: Vec<_> = avgs1
            .iter()
            .zip(avgs2.iter())
            .map(|(a1, a2)| *a1 - *a2)
            .collect();
        println!("amp,wav,ma,ref,diff,pos");
        for i in 0..len {
            println!(
                "{},{},{},{},{},{}",
                amp[i], wav[i], avgs1[i], avgs2[i], diffs[i], i
            );
        }
        /*
        using CSV, DataFrames, Gadfly
        df = CSV.read("ma_env.csv", DataFrame)
        plot(stack(df, Not([:pos])), x=:pos, y=:value, color=:variable, Geom.line)
        */
    }
}

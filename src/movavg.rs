pub struct MovAvg {
    obs: Vec<f32>,
    pos: usize,
    sum: f32,
}

impl MovAvg {
    pub fn new(window_size: usize, initial_avg: f32) -> Self {
        Self {
            obs: vec![initial_avg; window_size],
            pos: 0,
            sum: initial_avg * window_size as f32,
        }
    }

    pub fn update(&mut self, value: f32) {
        let prev = self.obs[self.pos];
        self.obs[self.pos] = value;
        self.sum = self.sum - prev + value;
        self.pos = (self.pos + 1) % self.obs.len();
    }

    pub fn avg(&self) -> f32 {
        self.sum / self.obs.len() as f32
    }

    pub fn update_and_get(&mut self, value: f32) -> f32 {
        self.update(value);
        self.avg()
    }
}

#[cfg(test)]
mod test {
    use super::MovAvg;

    #[test]
    fn test_movavg() {
        let mut ma = MovAvg::new(4, 10.0);
        assert_eq!(ma.avg(), 10.0);
        ma.update(50.0);
        assert_eq!(ma.avg(), 20.0);
    }
}

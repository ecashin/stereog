# Stereog

"Stereog" rhymes with "hairy dog."

Stereog is a [MIDI](https://vmpk.sourceforge.io/#MIDI_concepts)-controlled
stereo-preserving [granular synthesizer](https://www.soundonsound.com/techniques/granular-synthesis)
[LV2 plugin](https://lv2plug.in/).

It is experimental software,
which explains a few current oddities:

* It prints debug info to standard output
* It listens on all MIDI channels
* It resets the sampler when an A4 note is played

It is implemented in [Rust](https://www.rust-lang.org/)
and distributed under the MIT license.

During development, a Linux platform running [Ardour v6](https://ardour.org/),
the open-source digital audio workstation (DAW) was used for testing.
Is its virtual MIDI keyboard polyphonic?  Let me know.

## Principles of Operation

The plugin begins "listening" while in the "armed" state after initialization.

When the sound level passes a user-controlled threshold,
Stereog enters its "recording" mode.
When the sound level diminishes below another, lower threshold,
derived from the user-controlled threshold,
recording stops,
and the plugin switches to "playing" mode.

The captured sound serves as the source from grains
used to create the audio output.

Hitting an A4 note clears and arms the sampler.

## Future Work

Some intended developments are listed below.

* Use program-change messages to reset the sampler
* Listen to a specific MIDI channel that the user can select
* Try polyphony (with [VMPK](https://vmpk.sourceforge.io/)?)
* Try optimized builds
* Add "center" and "dispersal" controls
* Add number-of-grains control

## Build, Install, Test

An example command for building and installing appears below.
It depends on the wonderful [rsync](https://rsync.samba.org/) command.

    cargo build \
      && cp target/debug/libstereog.so stereog.lv2/ \
      && sudo rsync -av stereog.lv2 /usr/lib/lv2/

Tests are run as shown below.

    cargo test

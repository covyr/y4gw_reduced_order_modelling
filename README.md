Here I build a pipeline which constructs surrogate waveform models (offline), which can be used to generate waveforms (online) with considerably less computation time.

The offline stage consists of:
- Generating waveforms using a fiducial model across a parameter space
- Constructing a reduced basis from this large set of waveforms, picking the "most informative waveforms" and facilitating the optimal construction of any waveform in the parameter space from a small number of waveforms.
  (p1peline)
- Identifying the empirical time nodes from the reduced basis, picking the "most informative times", facilitating the optimal construction of waveforms by interpolating over time from a small number of time-series values.
  (p2peline)
- Training artificial neural networks to predict the waveform values at these empirical time nodes.
  (p3peline)

The online stage then consists of:
  > Predicting the waveform values at the empirical time nodes, given a set of parameters.
  > Interpolating across the time-series.

Some subject-specific information for this project:
- While this pipeline can be used to construct suurrogate waveform models generally, the specific waveforms in question throughout this project are gravitational waveforms produced by the merger of binary systems of black holes/neutron stars.
- Due to the geometric nature of general relativity, we work with the units c=G=M=1, adopting dimensions of t/M for time.
- We first decompose the full waveforms into their spin-weighted spherical harmonics to simplify the problem.
- Due to the highly oscillatory nature of the waveforms in question, we further decompose the modes into their amplitude and phase parts, which are much smoother and easier to model. These parts are modelled individually, and can be recombined to obtian the full waveforms.

Snapshots guide:

1) generator -- pretrained generative RNN
2) generator_benzene_ring -- generator that produces molecules with maximized number of benzene rings
3) generator_substituents -- generator that produces molecules with maximized number of substituents
4) generator_jak2_min -- generator that produces molecules inactive towards jak2 kinase
5) generator_jak2_max -- generator that produces molecules active towards jak2 kinase
6) generator_melting_temp_min -- generator that produces molecules with low melting temperatures
7) generator_melting_temp_max -- generator that produces molecules with high melting temperatures
8) generator_logP -- generator that produces molecules with logP property from 1.0 to 4.0
9) bio_oracle -- network for prediction bioactivity of jak2 kinase
10) logP_oracle -- network for logP prediction
11) melting_oracel -- network for melting temperature prediction

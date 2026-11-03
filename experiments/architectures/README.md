# architectures

**Question.** Do depth, width, the feedforward sublayer, and the learning rate
change the in-context result?

**Delta from base.** None beyond the swept axes; the feedforward width tracks
the residual width at four times.

**Axes.** n_layers {2, 4, 8} x d_model {128, 256} x feedforward {ffn, none} x
learning_rate {0.001, 0.0001} x 10 seeds.

**Training loss.** Every architecture is drawn; each panel prints the mean
training loss over the last 5,000 pretraining steps, averaged over seeds, so
an architecture that did not learn the pretraining task is visible at a glance.

**Feeds.** The appendix figure "Architecture and learning-rate sweep".

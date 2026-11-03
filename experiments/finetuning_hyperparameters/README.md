# finetuning_hyperparameters

**Question.** How do the replay proportion and the finetuning learning rate
trade off learning the fixed ordering against keeping the in-context strategy?

**Delta from base.** None beyond the swept axes.

**Axes.** replay_prop {0, 0.1, 0.2, 0.5} x learning_rate {0.00001, 0.0001,
0.001} x 10 seeds.

**Feeds.** The appendix figure "Replay during finetuning".

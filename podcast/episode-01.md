# Episode 01: Seventy-one percent, and the cards it still fumbles

This is the first episode for the third reinforcement learning model, the one we call Chain.

It began with a question. Does a network need a separate statistical model to play its action cards in the right order? Village before Smithy, Festival before the draw, what to put under a Throne Room. I built a turn planner to find out. It plays every option to the end of the turn, on a hundred and twenty-eight shuffles of the unseen cards. It graded more than fourteen thousand action decisions. The old champion network picked the best card about ninety-two percent of the time. But when the planner chose its actions instead, it did not win more games. Head to head it scored between forty-eight and fifty percent. So the player stays one network, with no chain module.

Instead, the model trained on. It started from the champion's last checkpoint and got new helper inputs. The curriculum had three stages. First its old league. Then a quarter of its games against the frozen champion. Then an open league with every earlier model. On top of that came a card school. In some games its first buys were forced onto cards the champion never tried, like Chapel and Mine. That way it learned to play them, while its buying stayed its own choice.

It went fast. The first stage passed after twenty iterations. The second passed after seventy-eight, at fifty-eight percent against the champion. The goal was fifty-five percent head to head, on a thousand fresh games. It first reached that at iteration three hundred and seventy-five. After twenty-eight hours and more than two thousand three hundred iterations, the best check on a thousand fresh games was seventy-one percent against the champion. That is the best Dominion model of the project so far.

Then, at about one o'clock today, the laptop ran out of memory. The system killed the terminal, and the training with it. Nothing was lost, because the last checkpoint was saved a minute earlier. Training now runs as a background service, so it is no longer tied to a terminal.

You asked which cards it loses against. So I played sixty-four thousand games. That was the current model against eight opponents, from both seats, on random kingdoms. It wins ninety-seven percent against Big Money. Against the champion it wins about seventy percent, and against the earlier networks about eighty.

The answer is that there is no card it simply cannot manage. The worst cards lower its win rate by three or four points when they are in the kingdom. Library and Moat do that, but they do the same to the old champion. That looks like the game itself, not a weakness of the model.

The sharper test was to play the old champion on exactly the same games, and compare card by card. There the card school shows up clearly. With Chapel in the kingdom the new model gains ten points over the champion. Village gains five, and Festival four. Those are the cards it was forced to practise.

And there is a small group where it is still weaker than the champion. Sentry and Gardens cost about five points each. Militia costs three. Witch, Harbinger, Workshop, Vassal, Mine and Bureaucrat cost one to two points.

So a new run is training now. It continues from the last checkpoint, with the same league and the same card school. In three games out of ten it gets a drill. The kingdom holds one of ten weak cards. Sometimes the model is forced to buy it. Sometimes its opponent is, so it learns to play against it. To keep what it already knows, the learning rate is a third lower.

Next, I watch the goal checks every twenty-five iterations. If the score against the champion drops below seventy, I stop. When it has trained long enough, I replay the same sixty-four thousand games, and we will see whether those cards closed the gap.

# Cardboard drawer reward lessons

## 15 cm drawer opening experiments

In the single-arm cardboard cabinet task, opening the inner box beyond roughly
10 cm was harder than reaching the handle/slot target or opening the drawer a
short distance.

Observed failure modes:

- If the outer cardboard box is too light, the policy can get reward by moving
  the whole outer box instead of cleanly pulling the inner box.
- A strong outer-box movement penalty alone can make the policy conservative:
  it reaches the target and starts opening, but avoids larger pulling actions.
- Target-reaching reward alone can make the policy stop near the green target
  instead of continuing to pull the drawer.

What worked for the 15 cm diagnostic run:

- Make the blue outer box heavier than the inner box.
- Keep the multiplicative outer-box shift penalty.
- Add extra shaping only after about 10 cm of opening:
  - late open reward toward a larger reward distance,
  - and a pull-lead TCP target ahead of the green target.

Current design decision:

- Do not keep the 10 cm+ special shaping in the main task.
- Treat 12 cm of inner-box opening as success for the simpler drawer task.
- Keep the above findings as diagnostic knowledge in case the 15 cm target is
  reintroduced later.

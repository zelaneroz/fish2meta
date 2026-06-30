# Does Learning Transfer During Prism Adaptation?

This repository contains the data and Python analysis for a student research project on **whether the cerebellum transfers motor learning between different hand movements during prism adaptation**. 

*This project was conducted as part of UWC ISAK Japan's CAS (Creativity, Activity, Service) Project requirements. The IB Diploma Program's CAS component requires students to engage in extracurriculars over 18 months, balancing the rigorous academics with real-world learning, personal growth, and community service. UWC ISAK Japan has a unique additional requirement to CAS called the 'CAS Project' where aside from the regular clubs and extracurriculars, students are required to embark on an independent long-term project of their choice, with the guidance of a faculty member.*

> View `Poster.png` in this repository for the final poster

> **Main finding:** adaptation and post-adaptation error correction were specific to the movement being trained. Learning an overhand throw did not automatically transfer to an underhand throw, or vice versa. This suggests that the cerebellum can maintain partly independent error-correction processes for different movements, even when they pursue the same goal.

![Mean displacement across the first experimental condition](results/figures/final_final_plots/mean%281-2%29.png)

## Research question

The cerebellum compares intended and observed movement, then adjusts motor commands to reduce error over repeated attempts. Prism glasses create a controlled visual shift, making this correction process directly observable: participants initially miss a target, gradually compensate for the shift, and then show a temporary error in the opposite direction when the glasses are removed. This final error is known as the **post-effect**.

We asked:

**When a person adapts one throwing movement while wearing prism glasses, does that learning transfer to a different throwing movement?**

The project tested two related hypotheses:

1. Prism adaptation learned with one hand movement would not transfer fully to another movement.
2. Overhand and underhand throws would retain independent adaptation and post-effect characteristics.

## Experimental design

Participants stood 2.95 metres from a whiteboard and threw a ball at a marked target. The landing position was recorded as horizontal displacement from the target, in centimetres. During adaptation trials, prism glasses shifted the participant's visual field by approximately 17 degrees.

Each protocol combined three phases:

- **Baseline:** overhand and underhand throws without prism glasses established each participant's normal accuracy and variability.
- **Prism adaptation:** participants repeatedly performed one or both movements while wearing prism glasses.
- **Post-effect:** participants repeated the movements without the glasses so we could measure how the learned correction decayed.

The team designed four protocol iterations and coordinated experiments involving more than 40 participants across protocol development and data collection. The final conditions compared transfer between overhand and underhand throwing and changed the order in which the movements were adapted to separate movement-specific learning from order effects.

## Analysis

The analysis pipeline was developed in Python using NumPy, SciPy, and Matplotlib. It:

1. extracted participant-level displacement measurements from the experimental records;
2. separated trials into baseline, adaptation, and post-effect phases;
3. cleaned and aggregated the measurements by participant and protocol;
4. calculated baseline precision from the spread of throws; and
5. fitted exponential decay curves to estimate how quickly adaptation and post-effects changed over successive throws.

The fitted model was:

$$f(t) = a - be^{-t/AC}$$

where $AC$ is the adaptation constant measured in number of throws. Smaller values indicate faster error correction.

## Results

The experiments showed a consistent movement-specific pattern:

- Post-effects were strongest for the movement trained under prism displacement.
- Training an overhand throw did not produce an equivalent post-effect in an underhand throw.
- When both movements were adapted, their learning curves and post-effects decayed independently.
- The first adaptation sequence sometimes produced a larger effect than the second, indicating a possible order effect that warrants further study.

Together, these observations support the conclusion that **overhand and underhand throws use separable error-correction processes during prism adaptation**. The result does not imply that all motor learning is movement-specific; it shows that transfer was limited under these protocols and for these two throwing movements.

## Repository contents

| Path | Description |
| --- | --- |
| `data/raw/` | Trial-level horizontal-displacement data from four protocol iterations |
| `data/interim/` | Intermediate data retained from the original analysis |
| `analysis/experiments/` | Participant and aggregate analyses for each experimental dataset |
| `analysis/exploratory/` | Exploratory curve fitting and visualization scripts |
| `results/figures/` | Generated participant, mean, and median plots |
| `results/summary.md` | Summary tables for baseline precision and adaptation constants |
| `archive/project-goldfish/` | Legacy reference material kept separate from the active analysis |

The scripts are preserved as research analysis snapshots rather than a packaged Python application. Run them from the repository root so their relative data and output paths resolve correctly.

```bash
python -m pip install -r requirements.txt
python analysis/experiments/experiment_1.py
```

## Project and contributions

This project was completed by a four-person team from UWC ISAK Japan as visiting researchers at the **Neural Cybernetics Laboratory, Chubu University**, through the 2022–2023 CAS program. It culminated in a scientific poster and oral presentation.

**Student researchers:** Luka Adamović, Zelan Eroz Espanto, Lison Hébert, and Alessandro Salvetti<br>
**Advisor:** Ruben Pinzon

Zelan Eroz Espanto led the Python-based data extraction, cleaning, curve fitting, and visualization. He also contributed to hypothesis development, protocol design, experimental coordination, interpretation of the results, and refinement of the research strategy with teammates and advisors.

## References

1. Martin, T. A., Keating, J. G., Goodkin, H. P., Bastian, A. J., & Thach, W. T. (1996). Throwing while looking through prisms: focal olivocerebellar lesions impair adaptation. *Brain, 119*(4), 1183–1198. https://doi.org/10.1093/brain/119.4.1183
2. Martin, T. A., Greger, B. E., Norris, S. A., & Thach, W. T. (2001). Dynamic coordination of body parts during prism adaptation. *Journal of Neurophysiology, 85*(4), 1685–1694. https://doi.org/10.1152/jn.2001.85.4.1685

## Acknowledgements

We thank Yutaka Hirata and the researchers at the Neural Cybernetics Laboratory for their guidance, facilities, and introduction to prism adaptation and motor learning. This research was supported by the UWC ISAK Japan CAS program.
 
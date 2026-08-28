# ohw26_proj_BoatPhone

**Folder Structure**

* `contributor_folders` Each contributor has their own folder and pushes their work here during the week, preventing merge conflicts.
* `final_notebooks` Final versions of the team's project notebooks live here.
* `scripts` Shared scripts or functions are added here.
* `data` Shared datasets are shared here.

## Start here

Three notebooks in `final_notebooks/` are the reproducible record of the whole project. Each is
written for somebody who has never seen this repository, each runs top to bottom, and each
states plainly what its pipeline does **and does not** establish. Read them in this order:

| Notebook | What it covers |
|---|---|
| **`optical_acquisition_pipeline.ipynb`** | Finding, screening and paying for the satellite photographs: one study box, three gates, 552 candidate scenes narrowed to 26 delivered ones — and how not to waste a monthly imagery budget that does not roll over. |
| **`vessel_detection_pipeline.ipynb`** | Finding boats in those 26 pictures. Mostly a record of what *failed*: four approaches were tried and abandoned, each for a measured reason, and the shipped detector is simpler than the one that scored best. |
| **`acoustic_pipeline.ipynb`** | The hydrophone half: getting an ONC token, downloading the recordings, computing band levels, counting vessel passages, and every figure. |

Expensive steps are switched off by default in all three — they print the command they would
have run and then read results already in the repository, so the whole pipeline is legible in a
few minutes without a six-hour download or a satellite order. Points where the optical and
acoustic halves meet are flagged in both directions; search any notebook for `HANDOFF`.

Approaches that were tried and retired live in `superseded/`, which is kept on disk but out of
git; its README records what each file was and the measurement that retired it.

## Project Name

Satellite-derived vessel distance, speed, and orientation relative to the Folger Deep hydrophone array improve the estimation of acoustic ranging for small-scale vessel noise.

## One-line Description

This project aims to improve the estimation of small-scale vessel noise detection ranging for the Folger Deep hydrophone array by calibrating acoustic profiles to optically measured distance, speed, and orientation from satellite-derived small-scale vessel detections.

## Collaborators

| Name                | Role                |
|---------------------|---------------------|
| Neve Foreman        | Vessel detection model development
| Malachy McCaffrey   | Optical imagery pipeline development
| Isaac Guld          | Acoustic data pipeline development



## Planning

* Initial idea: Cross-calibrating optical vessel detection against passive acoustics at Folger Deep
* Ideation Slide: https://docs.google.com/presentation/d/1_KLEDpLLvtKpH3awDlZRAiOKuHzbEti4CWmhEykuCG8/edit?slide=id.g3f85357d4e2_21_0#slide=id.g3f85357d4e2_21_0
* Slack channel: ohw26_proj_BoatPhone
* Final presentation: https://docs.google.com/presentation/d/1LnokajlD4dj0683nHa6zb9H3QxUS-2aq0sLbgzFS5yQ/edit?usp=sharing

## Background

Satellites see every boat but only for a fraction of a second per day; the hydrophone listens continuously but can't count. Use each to fix the other's blind spot.

## Goals

1) Search for, screen, and select PlanetScope imagery scenes for the vessel detection model.
2) Augment open-source machine learning vessel detection model to process selected PlanetScope scenes.
3) machine learning acoustic vessel noise signature model something something something
4) Integrate satellite / acoustic data ...
   
* Create a model that uses hydrophone data and prior knowledge of vessels detected from optical satellite imagery to create continuous estimates of the number of boats, and potential distinguish between larger or smaller boats (those without AIS)

## Datasets
* Acoustic data from Folger Deep hydrophone array
* PlanetScope 3m optical imagery

## Workflow/Roadmap

## Results/Findings

## Lessons Learned

## References


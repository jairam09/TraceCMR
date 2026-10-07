# Datasets

No data is committed to this repository. Download each dataset yourself into `data/<name>/` (the `data/` folder is ignored by git). Record the license and citation here the day you download it.

| Dataset | Species / sequence | Labels | Access | License | Status |
| --- | --- | --- | --- | --- | --- |
| ACDC | Human, short-axis cine | RV, LV myocardium, LV cavity at ED and ES | Register at https://acdc.creatis.insa-lyon.fr/ | Check terms on site before redistributing | To download |
| EMIDEC | Human, short-axis LGE | Myocardium, infarct, MVO | http://emidec.com/ | CC-BY-NC-SA | To download |
| M&Ms | Human, short-axis cine, multi-vendor | LV, myocardium, RV | Register on the challenge site | Check terms | Optional |
| UCL open mouse cardiac MRI (Shah, Steeden et al., ISMRM 2026) | Mouse, cine | See authors | Emailed authors to ask about access | Unknown | Requested |

## Label conventions

| Dataset | Native labels | Loader `label_map` |
| --- | --- | --- |
| ACDC | 0 background, 1 RV cavity, 2 myocardium, 3 LV cavity | `"acdc"` |
| EMIDEC | To be filled in when downloaded | To add |

## Citations

- ACDC: Bernard O, et al. Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: is the problem solved? IEEE Trans Med Imaging. 2018;37(11):2514-2525. doi:10.1109/TMI.2018.2837502
- EMIDEC: Lalande A, et al. Emidec: a database usable for the automatic evaluation of myocardial infarction from delayed-enhancement cardiac MRI. Data. 2020;5(4):89. doi:10.3390/data5040089
- UCL mouse dataset: Shah W, Stuckey D, Yao T, Wrobel M, Campbell R, Muthurangu V, Steeden J. Open-Access Mouse Cardiac MRI Dataset and Segmentation Model. ISMRM 2026, #507-04-004.

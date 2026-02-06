### InSpIRe - Interdependent Spatial Image Analysis for Regional assignment

<br />

The _InSpIRe_ Python package provides a computational framework that introduces sample interdependent probabilistic mapping used for regional assignment of regions of interest based on specific features.

The package is comprised of different components each of which is featured seperately in different code vignettes to examplify basic functionality. This includes:

* V1: Preparation of MIR imaging data for probabilistic mapping
* V2: Cohort-wide probabilistic mapping
* V3: Reference-based probabilistic mapping
* V4: Benchmarking of probabilistic maps
* V5: Multimodal correlation of results with MSI data

### Data availability
An example dataset including MIR imaging raw data and a brightfield teaching image for correlation with MSI data, as well as all intermediate results that can be produced using the different vignettes is provided in the folder example_data_set under the following folder structure:

example_data_set/
├── benchmarking_example_ID5a_Rec1/
│       ├── 1656_unprocessed_co-reg_ID5a.npy
│       ├── annotation_Rec1_ID5a.csv
│       ├── HE_40proz_ID5a.tif
│       └── HotSpotMap_cohortwide_tumor_1proz_ID5a.npy
│
├── feature_selection_prior_knowledge/
│       ├── features/
│       │     └── features_tumor.csv
│       └── rois/
│             └── Annotations_ID18.png
│
├── masked_projection_imgs/
│       ├── maskedProjectionImg_ID5a.npy
│       ├── maskedProjectionImg_ID18.npy
│       └── maskedProjectionImg_ID32.npy
│
├── MIRimaging_raw_data/
│       ├── IRIraw_ID5a.fsm
│       ├── IRIraw_ID18.fsm
│       └── IRIraw_ID32.fsm
│
└── multimodal_correlation/
│        ├── 1656_unprocessed_co-reg_ID32.npy
│        ├── HotSpotMap_cohortwide_tumor_1proz_ID32.npy
│        ├── HotSpotMap_sizeteachingimg_ID32.tif
│        └── teaching_ID32-9-30_MIRMSI.tif
│
└── probability_maps/
         ├── cohort_wide_mapping/
         │     ├── HotSpotMap_cohortwide_tumor_1proz_ID5a.npy
         │     ├── HotSpotMap_cohortwide_tumor_1proz_ID18.npy
         │     └── HotSpotMap_cohortwide_tumor_1proz_ID32.npy
         └── reference_based_mapping/
               ├── maskedProjectionImg_ID25.npy
               └── HotSpotMap_reference_based_tumor_ID25.npy

### Citing _InSpIRe_
Please cite the associated published article Rittel et al., 2026, Advanced Science.

### Installation
Download zip-file, extract and open directory. install the package manually.

´´´bash
pip install .
´´´



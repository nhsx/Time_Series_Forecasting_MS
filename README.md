# Time Series Forecasting - Model Comparison

## NHS England Digital Analytics and Research Team 

### About the Project

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)

This repository holds code for the forecasting comparison work - a short term project developed by [Milan Storey](https://github.com/MilanStorey) whilst on DDaT placement with DART in Nov/Dec 2023

_**Note:** Only public or fake data are shared in this repository._

### Project Stucture

The main code is found in the `src` folder of the repository.

```
.
├── docs                    # Documentation
├── src                     # Source files
├── .gitignore
├── mkdocs.yml              # Documentation page   
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENCE
├── OPEN_CODE_CHECKLIST.md
├── README.md
└── requirements.txt
```

### Getting Started

#### Installation

To get a local copy up and running follow these simple steps.

To clone the repo:

`git clone https://github.com/nhsx/Time_Series_Forecasting_MS

To create a suitable environment:
- ```python3.8 -m venv ~/venv/py3.8```
- `source venv/bin/activate`
- `pip install -r requirements.txt`

You can serve the documentation locally using `mkdocs serve`.

### Usage

```
src
├── ACF+PACF                 # Contains code to determine order of SARIMA model
├── Data                     # Contains data Files
├── Function                 # Contains the functions used in the main pipeline
├── Model Selection          # Contains the main pipeline where a model is selected   
├── Models                   # Contains the models
├── Prophet Scripts          # Contains a script to run the prophet model and the prophet model outputs
├── __init__.py
```

#### step 1: Run the acf/pacf script.

Determine the order of the SARIMA model by running the relevant script in the ACF+PACF folder. Guidance on how to interpret the resulting plots is provided in the mkdocs documentation page.

#### step 2: Run the prophet script.

Note the prophet function is only currently compatible with python <3.9, so this script must be run in a suitable environment.

#### step 3: Run the model selection script.

See mkdocs documentation page for what the outputs of this script look like, aswell as guidance on what parameters must be set prior to running.

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

### License

Unless stated otherwise, the codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENSE](./LICENSE) for more information._

The documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government 3.0][ogl] licence.

[mit]: LICENCE
[copyright]: http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

### Another Forecasting resource

For a simpler look into Forecasting with some beginner workbooks to go through, please see this repo;
_[Forecasting_1](https://github.com/nhsx/Forecasting/tree/main)._


### Contact

Primary Contact is Milan Storey (Milan.Storey1@nhs.net). Supervisor is Paul Carroll (Paul.Carroll9@nhs.net).

To find out more about the [Analytics Unit](https://www.nhsx.nhs.uk/key-tools-and-info/nhsx-analytics-unit/) visit our [project website](https://nhsx.github.io/AnalyticsUnit/projects.html) or get in touch at [england.tdau@nhs.net](mailto:england.tdau@nhs.net).

<!-- ### Acknowledgements -->

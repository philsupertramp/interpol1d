# Interpolation methods implemented in Python using numpy

Two methods are implemented and available at `src/interpol1d.py`:
- Newton-Polynomial-Interpolation `polinterpol`
- Cubic-Natural-Spline Interpolation `splineinterpol`

For more see the provided (german) [notebook](/interpolation.ipynb)

## Usage
Clone the repo:  
`git clone git@github.com:philsupertramp/interpol1d`

Run the tests:  
`python -m pytest src/test_interpol1d.py`

or with coverage:  
`coverage run -m pytest src/test_interpol1d.py; coverage report`

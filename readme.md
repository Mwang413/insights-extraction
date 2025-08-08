## Setup

1. Create and activate a virtual environment:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Install the Jupyter kernel:
   ```sh
   python -m ipykernel install --user --name insights-extraction
   ```

4. Open the notebook and select the `insights-extraction` kernel.


# To install requirements with pip
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name insights-extraction
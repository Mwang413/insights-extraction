## Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name insights-extraction


### Further work:

Outside of what I mentioned in markdown of the main "report.ipynb"
I would like to:

# build a chat functionality
I would like to have 'product identification' as an input to help track product's performance over time well'
If this can't be generated, I would create a object recognition pipeline to extract these through scraping instagram.

This may involve a RAG based Q&A

# build out a dashboard
make the plots live in a dashboard/front-end app
make UI/UX much cleaner so users can:
- drop the file
- asks top questsions
- select top categories for displaying
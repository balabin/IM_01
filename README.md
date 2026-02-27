# Reproducing the calculations

1. Inspect file Deploy.py and change the number of batches if desired.
2. Run file Deploy.py. This will create subfolders 'production_XXX'.
3. Run file run.sh. This will start all batch jobs.
4. After the batch jobs complete, run file Assemble.py. This will assemble results of individual runs into file assembled.csv.gz.
5. Start Jupyter and load notebook View.ipynb. Run the notebook. Select the accuracy metric to view (ROC AUC or enrichment).
6. Please send comments and/or bug reports, if any, to ibalabin@avicenna-bio.com.

File requirements.in lists the Python packages used for the calculations and visualization. There are no specific version requirements; any reasonably recent version of Python and packages will do.

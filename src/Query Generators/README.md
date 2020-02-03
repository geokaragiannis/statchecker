# FactChecking_IEA
## Query Generation

This code takes outputs from the classifiers (lists of files,row indices, years and formulas) and bruteforces the values to see if it matches a true value. A query is then generated.
We assume that all generic variables come from the same tab. (This is not always the case. I'll check about how good of an assumption it is.)


## Installation

```bash
pip3 install openpyxl
```

## Usage

Outputs of classifiers are stored in a dictionary as such:

```python

classifier_output={'files': ['files/World_Output_2018_NPS.xlsx'],
 'row_indices': ['TPEDrenew', 'PGElecDemand', 'Prod_Steel'],
 'years': [2040, 2016, 2017],
 'formulas': ['a', 'a+b', 'POWER(a/b,1/(y2-y1))-1']}
```


A dictionary is used to translate from formula to SQL query.

```python
{

"a": "(a)", 
"a+b": "SELECT (a) + (b) ;", 
"POWER(a/b,1/(y2-y1))-1": "SELECT(SELECT POWER((SELECT ((a)*1.0/(b))) ,(SELECT 1/(SELECT (y2-y1)))))-1;"
}
```

A true value is specified.

```python
true_value=0.030407493463541435
```

Run:

```python

python3 main.py
```

We get 2 queries:

```python
'Query: SELECT(SELECT POWER((SELECT ((SELECT 2017 FROM files/World_Output_2018_NPS.xlsx!!PG WHERE UVN=='PGElecDemand';)*1.0/(SELECT 2016 FROM files/World_Output_2018_NPS.xlsx!!PG WHERE UVN=='PGElecDemand';))) ,(SELECT 1/(SELECT (2017-2016)))))-1;'


'Query with values: SELECT(SELECT POWER((SELECT ((22209.16547315)*1.0/(21553.76937186047))) ,(SELECT 1/(SELECT (2017-2016)))))-1;'
```


The first query is the full nested query where each generic variables is replaced by a respective query.
The second query is a query given the values of the generic variables.



## TODO
- [ ] Need to add logging and comments to the code
- [ ] Need to add fact checking for near values
- [ ] Need to add more formula to queries


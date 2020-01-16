import requests
import pprint
import pandas as pd
from pandas.io.json import json_normalize

headers = {"Authorization": "token ce8f0a7f1e35efaae2e53372cb9db3cc370983a4"}

def run_query(query):  # A simple function to use requests.post to make the API call. Note the json= section.
    request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))


query = """
{{
   search(query: "{queryString}", type: REPOSITORY, first: {maxItems}) {{
     # repositoryCount
     edges {{
       node {{
         ... on Repository {{
           # name
           url
           # pullRequests {{ totalCount }}
           # openpullRequests: pullRequests(states:OPEN) {{totalCount}}
           # closedPullRequests: pullRequests(states:CLOSED) {{totalCount}}
           # mergedPullRequests: pullRequests(states:MERGED) {{totalCount}}
           # forks {{ totalCount }}
           # commitComments {{ totalCount }}
           # mentionableUsers {{ totalCount }}
           # assignableUsers {{ totalCount }}
           # issues {{ totalCount }}
           # totalIssues: issues {{totalIssues: totalCount}}
           # openIssues: issues(states:[OPEN]) {{openIssues: totalCount}}
           # languages {{ totalCount }}
           # releases {{ totalCount }}
           # watchers {{ totalCount }}
           # stargazers {{ totalCount }}
           # master: object(expression:"master") {{
           #   ... on Commit {{
           #     history(since:  "2019-04-30T00:00:00Z") {{
           #       # edges {{
           #       #   node {{
           #       #     author {{ email }}
           #       #   }}
           #       # }}
           #       commits: totalCount
           #     }}
           #   }}
           # }}
         }}
       }}
    }}
  }}
}}
"""
variables = {
   'queryString' : 'is:public archived:false stars:3500..5000 size:>=3000 created:>=2014-01-01',
   'maxItems' : '60'
    # stars:10000..20000 repo:tensorflow/tensorflow language:C++ vue in:name pushed:>=2013-02-01
}

result = run_query(query.format(**variables))

# print(result)
pprint.pprint(result)

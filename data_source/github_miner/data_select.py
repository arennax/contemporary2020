# # data1.json
#
#
#
# ```
# {
#   search(query: "is:public archived:False stars:1000..3000 size:>=3000 created:>=2012-01-01", type: REPOSITORY, first: 100) {
#     edges {
#       node {
#         ... on Repository {
#           url
#         }
#       }
#     }
#   }
# }
#
# ```
# # data2.json
#
# ```
# {
#   search(query: "is:public archived:False stars:3500..5000 size:>=3000 created:>=2014-01-01", type: REPOSITORY, first: 100) {
#     edges {
#       node {
#         ... on Repository {
#           url
#         }
#       }
#     }
#   }
# }
#
# ```
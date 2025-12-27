# get a query. return time ranges for filtering chunks
# llm return a json:
{
  "analysis": {
    "is_temporal_query": true,
    "intent_type": "comparison / current_status / specific_event / general_knowledge"
  },
  "hard_filter_params": {
    "requires_filtering": true,
    "filter_logic": "OR",
    "time_ranges": [
      {
        "start_date": "ISO_8601",
        "end_date": "ISO_8601",
        "label": "string description",
        "granularity": "day/week/month/year"
      }
    ]
  },
}

# for example:
{
  "analysis": { "is_temporal_query": true, "intent_type": "comparison" },
  "hard_filter_params": {
    "requires_filtering": true,
    "filter_logic": "OR",
    "time_ranges": [
      {
        "start_date": "2023-10-01T00:00:00Z",
        "end_date": "2023-12-31T23:59:59Z",
        "label": "Q4 2023",
        "granularity": "quarter"
      },
      {
        "start_date": "2024-10-01T00:00:00Z",
        "end_date": "2025-12-31T23:59:59Z",
        "label": "Q4 2023",
        "granularity": "year"
      }
    ]
  }
}
import json

filepath = r'c:\Users\USER\Desktop\IR\IR2026\excercise4\outputs\analysis\recency_runs_flat.jsonl'

# Evolution keywords to identify comparative/change queries
evolution_keywords = [ 'budget']

with open(filepath, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        
        # Filter conditions
        k = data.get('k')
        representation = data.get('pipeline_representation')
        query = data.get('query', '').lower()
        chunking = data.get('pipeline_chunking')
        
        # Check if query contains evolution keywords
        is_evolution_query = any(keyword in query for keyword in evolution_keywords)
        
        # Check if matches criteria: evolution query, dense representation, k=5 or k=10
        if is_evolution_query and k in [5, 10]:
            print('=' * 100)
            print(f'Query: {data.get("query")}')
            print()
            print(f'Pipeline: chunking={data.get("pipeline_chunking")}, representation={representation}')
            print(f'K: {k}')
            print()
            print(f'Answer ({data.get("answer_length")} chars):')
            print(data.get('answer'))
            print('=' * 100)
            print()

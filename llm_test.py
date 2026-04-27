# non stream

from llm_proxy import create_simple_proxy

import os
tracing_info = {"asset_id": "3363", "trans_id": "", "user_id": "", "user_type": "", "ext_info": {}}
# load local config for dev
llm = create_simple_proxy('{query}', 'OpenAI_4o-2024-11-20_LexisPlusAIUS_3363_prod',
                          max_tokens=200,
                          temperature=0.5,
                          retry=1,
                          timeout=180,
                          reasoning_effort="low")
query = """
who are you?, where is US?
"""
# Setting answer_only=False
llm_predict_result = llm.predict(query=query, tracing_info=tracing_info)
print(f"Answer: {llm_predict_result}")


# stream
import os
tracing_info = {"asset_id": "3363", "trans_id": "", "user_id": "", "user_type": "", "ext_info": {}}
# load local config for dev
llm = create_simple_proxy('{query}', 'OpenAI_4o-2024-11-20_LexisPlusAIUS_3363_prod',
                          max_tokens=200,
                          temperature=0,
                          retry=1,
                          timeout=180)
query = """
who are you?, where is US?
"""
# Setting answer_only=False
llm_predict_result = llm.predict(query=query, stream=True, tracing_info=tracing_info)

for i in llm_predict_result:
    print(f"Answer: {i}")
 
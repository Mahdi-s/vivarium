#!/bin/bash
echo "=== Verifying New Features ==="

# 1. Settings module
echo -n "1. Settings module: "
python -c "from aam.settings import settings; print('✓')" || echo "✗"

# 2. Scientific report
echo -n "2. Scientific report generator: "
python -c "from aam.analytics.reporting import ScientificReportGenerator; print('✓')" || echo "✗"

# 3. Sparse capture config
echo -n "3. Sparse capture config: "
python -c "from aam.interpretability import CaptureConfig; c=CaptureConfig.from_dict({'layers':[0],'components':['resid'],'trigger_actions':['test'],'layer_sample_rate':0.5}); print('✓')" || echo "✗"

# 4. CoT indexing
echo -n "4. CoT indexing: "
python -c "from aam.interpretability import CaptureContext, CaptureConfig; c=CaptureConfig(layers=[0],components=['resid'],trigger_actions=['test']); ctx=CaptureContext(output_dir='/tmp',config=c); print('✓' if hasattr(ctx,'mark_cot_region') else '✗')" || echo "✗"

# 5. Dual-stack detection
echo -n "5. Dual-stackon -c "from aam.llm_gateway import select_local_gateway; print('✓' if 'scientific_mode' in select_local_gateway.__code__.co_varnames else '✗')" || echo "✗"

# 6. Probe dataset diversity
echo -n "6. Probe dataset diversity: "
python -c "import json; items=[json.loads(l) for l in open('experiments/olmo_conformity/datasets/candidates/social_probe_train.jsonl')]; prefixes=[i.get('source',{}).get('prefix_template') for i in items if i.get('label')==1]; print('✓' if len(set(prefixes)) >= 5 else '✗')" || echo "✗"

echo "=== Verification Complete ==="

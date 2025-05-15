mkdir -p exp/all_requests/task_outputs
for TASK in T01LegalMDS T02LegalSOF T03MaterialSEG T04EduPAE T05EduFG T06HealthCNG T07ChemMDG T08BioPDF T09MedicalDR T10FinanceESG T11CyberRDG
do
    python data_scripts/create_task_output_requests.py \
        --data_file exp/data/${TASK}.jsonl \
        --task_config configs/tasks/${TASK}.yaml \
        --output_file exp/all_requests/task_outputs/${TASK}.jsonl
done
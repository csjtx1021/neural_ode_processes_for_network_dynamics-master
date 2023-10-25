

# generating test sets for testing sparsity
python generate_testset_for_sparsity.py

for dynamics_name in 'mutualistic_interaction_dynamics'
do
  for topo_type in 'all'
  do
    for bound_t_context in 0.5
    do
      for sparsity in 0.012 0.014 0.016 0.018 0.02 0.03 0.04 0.05 0.10
      do
        for rseed in 1
        do
          echo ${bound_t_context} ${dynamics_name} ${topo_type} ${rseed} ${sparsity}
          python NDP4ND_testSparsity.py --sparsity=${sparsity} --is_uncertainty --is_determinate --dynamics_name=${dynamics_name} --topo_type=${topo_type} --x_dim=4 --seed=${rseed} --bound_t_context=${bound_t_context} --test --num_epochs=30 |tee results/NDP4ND_testSparisty_${dynamics_name}_${topo_type}_bound_t_context=${bound_t_context}.txt
        done
      done
    done
  done
done

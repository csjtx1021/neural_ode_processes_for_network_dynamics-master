
# generating test sets for testing noise
python generate_testset_withNoise.py

for dynamics_name in 'mutualistic_interaction_dynamics'
do
  for topo_type in 'all'
  do
    for bound_t_context in 0.5
    do
      for noise in 0.1 0.2 0.3 0.4 0.5
      do
        for rseed in 1
        do
          echo ${bound_t_context} ${dynamics_name} ${topo_type} ${rseed} ${noise}
          python NDP4ND_testNoise.py --noise=${noise} --is_uncertainty --is_determinate --dynamics_name=${dynamics_name} --topo_type=${topo_type} --x_dim=4 --seed=${rseed} --bound_t_context=${bound_t_context} --test --num_epochs=30 |tee results/NDP4ND_testNoise_${dynamics_name}_${topo_type}_bound_t_context=${bound_t_context}.txt
        done
      done
    done
  done
done

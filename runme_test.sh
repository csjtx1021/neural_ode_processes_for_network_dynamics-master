
# generating test sets
python generate_testsets.py

# testing
##-------------------------
for dynamics_name in 'mutualistic_interaction_dynamics'
do
  for topo_type in 'all'
  do
    for bound_t_context in 0.5
    do
      for rseed in 1
      do
        echo ${bound_t_context} ${dynamics_name} ${topo_type} ${rseed}
        python NDP4ND.py --is_uncertainty --is_determinate --dynamics_name=${dynamics_name} --topo_type=${topo_type} --x_dim=4 --seed=${rseed} --bound_t_context=${bound_t_context} --test --num_epochs=30 |tee results/NDP4ND_${dynamics_name}_${topo_type}_bound_t_context=${bound_t_context}.txt
      done
    done
  done
done

##-------------------------
for dynamics_name in 'phototaxis_dynamics'
do
  for topo_type in 'full_connected'
  do
    for bound_t_context in 0.5
    do
      for rseed in 1
      do
        echo ${bound_t_context} ${dynamics_name} ${topo_type} ${rseed}
        python NDP4ND.py --is_uncertainty --is_determinate --dynamics_name=${dynamics_name} --topo_type=${topo_type} --x_dim=4 --seed=${rseed} --bound_t_context=${bound_t_context} --test --num_epochs=200 |tee results/NDP4ND_${dynamics_name}_${topo_type}_bound_t_context=${bound_t_context}.txt
      done
    done
  done
done

##-------------------------
for dynamics_name in 'brain_FitzHugh_Nagumo_dynamics'
do
  for topo_type in 'power_law'
  do
    for bound_t_context in 0.5
    do
      for rseed in 1
      do
        echo ${bound_t_context} ${dynamics_name} ${topo_type} ${rseed}
        python NDP4ND.py --is_uncertainty --is_determinate --dynamics_name=${dynamics_name} --topo_type=${topo_type} --x_dim=4 --seed=${rseed} --bound_t_context=${bound_t_context} --test --num_epochs=200 |tee results/NDP4ND_${dynamics_name}_${topo_type}_bound_t_context=${bound_t_context}.txt
      done
    done
  done
done

##-------------------------
for dynamics_name in  'SIS_Individual_dynamics' 'SIR_Individal_dynamics' 'SEIS_Individual_dynamics'
do
  for topo_type in 'power_law'
  do
    for bound_t_context in 0.5
    do
      for rseed in 1
      do
        echo ${bound_t_context} ${dynamics_name} ${topo_type} ${rseed}
        python NDP4ND.py --is_uncertainty --is_determinate --dynamics_name=${dynamics_name} --topo_type=${topo_type} --x_dim=4 --seed=${rseed} --bound_t_context=${bound_t_context} --test --num_epochs=30 |tee results/NDP4ND_${dynamics_name}_${topo_type}_bound_t_context=${bound_t_context}.txt
      done
    done
  done
done

##-------------------------
# generating test sets for empirical systems
python generate_testset_for_sparsity.py

# testing for empirical systems
declare -A bound_t_context
#bound_t_context=( [6]=0.9 [10]=0.5 [11]=0.91 [20]=0.5 )
bound_t_context=( [6]=0.2 [10]=0.2 [11]=0.43 [20]=0.43 )

for dynamics_name in 'RealEpidemicData_mix'
do
  for topo_type in 'power_law'
  do
    for sparsity in 0.05 0.1 0.2 0.3 0.4 0.5
    do
        for time_steps in 6 20
        do
          for rseed in 1
          do
            echo ${bound_t_context[${time_steps}]} ${dynamics_name} ${topo_type} ${rseed}
            python NDP4ND.py --is_uncertainty --is_determinate --dynamics_name=${dynamics_name} --topo_type=${topo_type} --x_dim=4 --seed=${rseed} --bound_t_context=${bound_t_context[${time_steps}]} --num_graphs=2000 --test --num_epochs=60 --time_steps=${time_steps} --latent_dim=50 --hidden_dim=50 --is_sparsity --sparsity=${sparsity} |tee results/NDP4ND_${dynamics_name}_${topo_type}.txt
          done
        done
    done
  done
done





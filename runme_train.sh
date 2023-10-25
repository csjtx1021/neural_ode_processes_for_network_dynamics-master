
for dynamics_name in 'mutualistic_interaction_dynamics'
do
  for topo_type in 'all'
  do
        python NDP4ND.py --is_uncertainty --is_determinate --dynamics_name=${dynamics_name} --topo_type=${topo_type} --x_dim=4 --seed=0 --train --num_epochs=30 2>&1 |tee results/NDP4ND_${dynamics_name}_${topo_type}_train_log__.txt
  done
done

##-------------------------
for dynamics_name in 'phototaxis_dynamics'
do
  for topo_type in 'full_connected'
  do
        python NDP4ND.py --is_uncertainty --is_determinate --dynamics_name=${dynamics_name} --topo_type=${topo_type} --x_dim=4 --seed=0 --train --num_epochs=200 2>&1 |tee results/NDP4ND_${dynamics_name}_${topo_type}_train_log__.txt
  done
done

##-------------------------
for dynamics_name in 'brain_FitzHugh_Nagumo_dynamics'
do
  for topo_type in 'power_law'
  do
        python NDP4ND.py --is_uncertainty --is_determinate --dynamics_name=${dynamics_name} --topo_type=${topo_type} --x_dim=4 --seed=0 --train --num_epochs=200 2>&1 |tee results/NDP4ND_${dynamics_name}_${topo_type}_train_log__.txt
  done
done

##-------------------------
for dynamics_name in 'SIS_Individual_dynamics' 'SIR_Individual_dynamics' 'SEIS_Individual_dynamics'
do
  for topo_type in 'power_law'
  do
        python NDP4ND.py --is_uncertainty --is_determinate --dynamics_name=${dynamics_name} --topo_type=${topo_type} --x_dim=4 --seed=0 --train --num_epochs=30 2>&1 |tee results/NDP4ND_${dynamics_name}_${topo_type}_train_log__.txt
  done
done

##-------------------------
for dynamics_name in 'RealEpidemicData_mix'
do
  for topo_type in 'power_law'
  do
    python NDP4ND.py --is_uncertainty --is_determinate --dynamics_name=${dynamics_name} --topo_type=${topo_type} --x_dim=4 --seed=0 --train --num_graphs=1000 --num_epochs=60 --latent_dim=50 --hidden_dim=50 |tee results/NDP4ND_${dynamics_name}_${topo_type}_train_log_num_epochs=60__.txt
  done
done




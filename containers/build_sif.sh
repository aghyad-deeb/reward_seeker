# On a worker node
export TMPDIR=$LOCALDIR
singularity build $PROJECTDIR/containers/verl_vllm012.sif $PROJECTDIR/reward_seeker/verl_vllm012.def

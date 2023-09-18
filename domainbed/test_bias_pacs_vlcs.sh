for bias in 0.3 0.5 0.7 1.0
  do
    for env in 0 1 2 3
    do
      python3 -m domainbed.scripts.train\
             --data_dir=./domainbed/data/\
             --algorithm InterRM\
             --dataset PACS\
             --test_env $env \
             --hyper_bias $bias
      python3 -m domainbed.scripts.train\
             --data_dir=./domainbed/data/\
             --algorithm InterRMMMD\
             --dataset PACS\
             --test_env $env \
             --hyper_bias $bias
    done
done

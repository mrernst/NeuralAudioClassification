curl http://opihi.cs.uvic.ca/sound/genres.tar.gz -o ./data/genres.tar.gz
tar -zxvf ./data/genres.tar.gz
curl https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/train_filtered.txt -o ./data/train_filtered.txt
curl https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/valid_filtered.txt  -o ./data/valid_filtered.txt
curl https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/test_filtered.txt -o ./data/test_filtered.txt
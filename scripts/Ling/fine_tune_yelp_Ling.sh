set -x
cp configs/bert_yelp_Ling.config run.config
python fine_tune_bert.py
cp configs/cbert_yelp_Ling.config run.config
python fine_tune_cbert.py
#PYTHONPATH=$PROJECTPATH $PYTHON_HOME/python test_tools/yang_test_tool/cls_wd.py
python fine_tune_cbert_w_cls.py

.PHONY: proposal remove_proposal \
	train_test_split remove_train_test_split \
	sampled_dataset remove_sampled_dataset \
	full_dataset remove_full_dataset \
	remove_cached_dataset \
	results remove_results \
	model remove_model \
	final_report remove_final-report

## render proposal
proposal: report/proposal/proposal.pdf \
	report/proposal/proposal.html 

report/proposal/proposal.pdf:
	quarto render report/proposal/proposal.qmd --to pdf

report/proposal/proposal.html:
	quarto render report/proposal/proposal.qmd --to html

# remove proposal
remove_proposal:
	rm -f report/proposal/proposal.pdf \
			report/proposal/proposal.html
	rm -rf report/proposal/proposal_files

## generate list of email paths and target labels
train_test_split: data/full-dataset/test.csv \
	data/full-dataset/train.csv \
	data/sampled-dataset/sample-large.csv \
	data/sampled-dataset/sample-small.csv \

data/full-dataset/test.csv data/full-dataset/train.csv data/sampled-dataset/sample-large.csv data/sampled-dataset/sample-small.csv:
	python scripts/generate_email_list.py

# remove list of email paths and target labels
remove_train_test_split:
	rm -f data/full-dataset/test.csv \
			data/full-dataset/train.csv \
			data/sampled-dataset/sample-large.csv \
			data/sampled-dataset/sample-small.csv

## generate features for sample-small, sample-large datasets
sampled_dataset: data/sampled-dataset/raw/sample-small.parquet \
	data/sampled-dataset/raw/sample-large.parquet \
	data/sampled-dataset/processed/sample-small-partial.parquet \
	data/sampled-dataset/processed/sample-large-partial.parquet \
	data/sampled-dataset/processed/sample-small.parquet \
	data/sampled-dataset/processed/sample-large.parquet

data/sampled-dataset/raw/sample-small.parquet:
	python -W ignore scripts/build_metadata_df.py --dataset sample-small

data/sampled-dataset/raw/sample-large.parquet:
	python -W ignore scripts/build_metadata_df.py --dataset sample-large

data/sampled-dataset/processed/sample-small-partial.parquet:
	python -W ignore scripts/build_features_df.py --dataset sample-small --quick n

data/sampled-dataset/processed/sample-large-partial.parquet:
	python -W ignore scripts/build_features_df.py --dataset sample-large --quick n

data/sampled-dataset/processed/sample-small.parquet:
	python -W ignore scripts/build_features_df.py --dataset sample-small --quick y

data/sampled-dataset/processed/sample-large.parquet:
	python -W ignore scripts/build_features_df.py --dataset sample-large --quick y

# remove sample-small and sample-large datasets
remove_sampled_dataset:
	rm -f data/sampled-dataset/raw/sample-small.parquet \
			data/sampled-dataset/raw/sample-large.parquet \
			data/sampled-dataset/processed/sample-small.parquet \
			data/sampled-dataset/processed/sample-large.parquet

## generate features for train and test datasets
full_dataset: data/full-dataset/raw/train.parquet \
	data/full-dataset/raw/test.parquet \
	data/full-dataset/processed/train-partial.parquet \
	data/full-dataset/processed/test-partial.parquet \
	data/full-dataset/processed/train.parquet \
	data/full-dataset/processed/test.parquet

data/full-dataset/raw/train.parquet:
	python -W ignore scripts/build_metadata_df.py --dataset train

data/full-dataset/raw/test.parquet:
	python -W ignore scripts/build_metadata_df.py --dataset test

data/full-dataset/processed/train-partial.parquet:
	python -W ignore scripts/build_features_df.py --dataset train --quick n

data/full-dataset/processed/test-partial.parquet:
	python -W ignore scripts/build_features_df.py --dataset test --quick n

data/full-dataset/processed/train.parquet:
	python -W ignore scripts/build_features_df.py --dataset train --quick y

data/full-dataset/processed/test.parquet:
	python -W ignore scripts/build_features_df.py --dataset test --quick y

# remove train and test datasets
remove_full_dataset:
	rm -f data/full-dataset/raw/train.parquet \
			data/full-dataset/raw/test.parquet \
			data/full-dataset/processed/train.parquet \
			data/full-dataset/processed/test.parquet

## remove cached datasets containing online features (train/test-partial.parquet)
remove_cached_dataset:
	rm -f data/sampled-dataset/processed/sample-small-partial.parquet \
			data/sampled-dataset/processed/sample-large-partial.parquet \
			data/full-dataset/processed/train-partial.parquet \
			data/full-dataset/processed/test-partial.parquet

## model selection results
results: results/base_classifier_selection/cv_results.csv \
	results/base_classifier_selection/xgb_feature_importances.csv \
	results/base_classifier_selection/xgb_feature_importances.png \
	results/stacking_final_estimator/cv_results.csv \
	results/stacking_final_estimator/header_feat_importances.csv \
	results/stacking_final_estimator/header_feat_importances.png \
	results/stacking_final_estimator/body_feat_importances.csv \
	results/stacking_final_estimator/body_feat_importances.png \
	results/model_architecture_selection/cv_results.csv \
	results/BERT/BERT_classification_results.csv \
	results/BERT/BERT_sentiment_results.csv \
	results/BERT/email_count_per_topic_chart.png \
	results/final_results/classification_report.csv \
	results/final_results/cm_phishsense_test.png \
	results/final_results/cm_phishsense-v2_test.png \
	results/final_results/cm_phishsense-v2_train.png

results/base_classifier_selection/cv_results.csv results/base_classifier_selection/xgb_feature_importances.csv results/base_classifier_selection/xgb_feature_importances.png:
	python -W ignore scripts/base_classifier_selection.py \
		--train_data data/sampled-dataset/processed/sample-large.parquet \
		--results_to results/

results/stacking_final_estimator/cv_results.csv results/stacking_final_estimator/header_feat_importances.csv results/stacking_final_estimator/header_feat_importances.png results/stacking_final_estimator/body_feat_importances.csv results/stacking_final_estimator/body_feat_importances.png:
	python -W ignore scripts/stacking_final_estimator_selection.py \
		--train_data data/sampled-dataset/processed/sample-large.parquet \
		--results_to results/

results/model_architecture_selection/cv_results.csv:
	python -W ignore scripts/model_architecture_selection.py \
		--train_data data/sampled-dataset/processed/sample-large.parquet \
		--results_to results/

results/BERT/BERT_classification_results.csv results/BERT/BERT_sentiment_results.csv results/BERT/email_count_per_topic_chart.png:
	python -W ignore scripts/get_BERT_results.py \
		--train_data data/sampled-dataset/processed/sample-small.parquet \
		--results_to results/

results/final_results/classification_report.csv results/final_results/cm_phishsense_test.png results/final_results/cm_phishsense-v2_test.png results/final_results/cm_phishsense-v2_train.png:
	python -W ignore scripts/phishsense-v2_results.py \
		--train_data data/full-dataset/processed/train.parquet \
		--test_data data/full-dataset/processed/test.parquet \
		--phishsense_results results/final_results/phishsense_predictions.csv \
		--model_path model/phishsense-v2.pkl \
		--results_to results/

remove_results:
	rm -f results/base_classifier_selection/cv_results.csv \
			results/base_classifier_selection/xgb_feature_importances.csv \
			results/base_classifier_selection/xgb_feature_importances.png \
			results/stacking_final_estimator/cv_results.csv \
			results/stacking_final_estimator/header_feat_importances.csv \
			results/stacking_final_estimator/header_feat_importances.png \
			results/stacking_final_estimator/body_feat_importances.csv \
			results/stacking_final_estimator/body_feat_importances.png \
			results/model_architecture_selection/cv_results.csv \
			results/BERT/BERT_classification_results.csv \
			results/BERT/BERT_sentiment_results.csv \
			results/BERT/email_count_per_topic_chart.png \
			results/final_results/classification_report.csv \
			results/final_results/cm_phishsense_test.png \
			results/final_results/cm_phishsense-v2_test.png \
			results/final_results/cm_phishsense-v2_train.png

## train model
model: model/phishsense-v2.pkl

model/phishsense-v2.pkl:
	python -W ignore scripts/train_model.py \
		--train_data data/full-dataset/processed/train.parquet \
		--model_to model/ \
		--tuning n

# remove model
remove_model:
	rm -f model/phishsense-v2.pkl

## render report
final_report: report/final_report/final_report.html report/final_report/final_report.pdf

report/final_report/final_report.html report/final_report/final_report.pdf:
	quarto render report/final_report/final_report.qmd

remove_final_report:
	rm -f report/final_report/final_report.html \
		report/final_report/final_report.pdf
	rm -rf report/final_report/final_report_files

--- a/test_skitti.py
+++ b/test_skitti.py
@@ -69,9 +69,12 @@
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

- SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
- unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
- unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

* # For binary classification: Ground (0), Obstacle (1)
* binary_label_name = {0: "ground", 1: "obstacle"}
* # unique_label_for_hist_crop are the actual class values (0 and 1)
* # that fast_hist will build its confusion matrix on.
* # The number of classes for fast_hist will be dataset_config.num_classes (which is 2).
* unique_label_str = [binary_label_name[x] for x in sorted(binary_label_name.keys())]

  my_model = get_model_class(model_config['model_architecture'])(configs)

@@ -96,9 +99,8 @@

     label_mapping = dataset_config["label_mapping"]

- with open(dataset_config['label_mapping'], 'r') as stream:
-        mapfile = yaml.safe_load(stream)
-
- valid_labels = np.vectorize(mapfile['learning_map_inv'].**getitem**)

* # The valid_labels mapping from learning_map_inv is for original 19 classes to raw KITTI labels.
* # For binary ground/obstacle, this direct mapping is not applicable for submission to original benchmark.
* # We will save predicted labels as 0 (ground) or 1 (obstacle).

  SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])

@@ -153,7 +155,8 @@
val_pt_labs = raw_labels.cpu().detach().numpy()

                 if train_hypers.local_rank == 0:

-                    hist_list.append(fast_hist_crop(predict_labels, val_pt_labs, unique_label))

*                    # For binary classification, fast_hist directly gives the 2x2 confusion matrix
*                    hist_list.append(fast_hist(predict_labels, val_pt_labs, model_config.num_classes))
                       pbar.update(1)

           if train_hypers.local_rank == 0:

  @@ -203,15 +206,14 @@
  print('%s already exsist...' % (new_save_dir))
  pass

-                    test_pred_label = valid_labels(test_pred_label)
-                    test_pred_label = test_pred_label.astype(np.uint32)
-                    test_pred_label.tofile(new_save_dir)

*                    # Save binary labels (0 for ground, 1 for obstacle)
*                    test_pred_label_save = test_pred_label.astype(np.uint8) # Use uint8 for 0/1 values
*                    test_pred_label_save.tofile(new_save_dir)
                     pbar.update(1)

         if train_hypers.local_rank == 0:
             pbar.close()

-            # print('Predicted test labels are saved in %s. Need to be shifted to original label format before submitting to the Competition website.' % exp_dir)
-            # print('Remapping script can be found in semantic-kitti-api.')

*            print(f'Predicted binary (0:ground, 1:obstacle) test labels are saved in {exp_dir}')

if **name** == '**main**':
print(' '.join(sys.argv))

--- a/dataloader/dataset2.py
+++ b/dataloader/dataset2.py
@@ -35,7 +35,10 @@
{'car': 0, 'bicycle': 1, 'motorcycle': 2, 'truck': 3, 'other-vehicle': 4, 'person': 5, 'bicyclist': 6, 'motorcyclist': 7,
'road': 8, 'parking': 9, 'sidewalk': 10, 'other-ground': 11, 'building': 12, 'fence': 13, 'vegetation': 14, 'trunk': 15,
'terrain': 16, 'pole': 17, 'traffic-sign': 18}

- +Original instance classes for SemanticKITTI (learning IDs 1-8 for vehicles and people)
  """
  -instance_classes = [1, 2, 3, 4, 5, 6, 7, 8] # [2, 3, 4, 5, 7, 8, 10, 13, 14, 17, 20] #
  +# For binary classification, if polarmix is used to augment "obstacle" instances,
  +# instance_classes should refer to the new obstacle label ID.
  +instance_classes = [1] # New obstacle label ID is 1.
  Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3] # x3
  @register_dataset

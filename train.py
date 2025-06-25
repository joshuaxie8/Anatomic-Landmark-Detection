from __future__ import print_function, division
import torch
import time
import utils
import numpy as np
from tqdm import tqdm

def train_model(model, dataloaders, criterion, optimizer, config):
	since = time.time()

	losses = []
	# validation for every 5 epoches
	test_epoch = 5
	
	# Early stopping variables
	best_val_mre = float('inf')
	patience = 10
	patience_counter = 0
	
	for epoch in range(config.epochs):
		train_dev = []
		for phase in ['train']:
			model.train(True)  # Set model to training mode
			running_loss = 0.0
			# Iterate over data.
			lent = len(dataloaders[phase])
			pbar = tqdm(total=lent * config.batchSize)
			for ide in range(lent):
				data = dataloaders[phase][ide]
				
				inputs, labels = data['image'], data['landmarks']
				inputs = inputs.to(config.use_gpu)

				optimizer.zero_grad()
				# forward
				heatmaps = model(inputs)

				# loss calculation for one heatmap and two offset maps.
				loss = criterion(heatmaps[0], labels.detach().cpu())
				#~ # backward + optimize only if in training phase
				loss.backward()
				optimizer.step()

				if epoch%test_epoch == 0:
					# landmark prediction. The results are normalized to (0, 1)
					predicted_landmarks = utils.regression_voting(heatmaps, config.R2).cuda(config.use_gpu)
					# deviation calculation for all landmarks
					dev = utils.calculate_deviation(predicted_landmarks.detach(), labels.cuda(config.use_gpu).detach())
					train_dev.append(dev)

				running_loss += loss.item()
				pbar.update(config.batchSize)
			pbar.close()

			epoch_loss = running_loss / lent
			losses.append(epoch_loss)
			print('{} epoch: {} Loss: {}'.format(phase, epoch, epoch_loss))

		# validation
		if epoch%test_epoch == 0:
			# result statistics
			train_dev = torch.cat(train_dev, dim=0) * config.spacing
			train_SDR, train_SD, train_MRE = utils.get_statistical_results(train_dev, config)

			# MRE is the mean radial error, SDR is the the successful detection rate in five target radius (1mm, 2mm, 2.5mm, 3mm, 4mm)
			print("train_MRE(SD): %f(%f), SDR([1mm, 2mm, 2.5mm, 3mm, 4mm]): " % (torch.mean(train_MRE).detach().cpu().numpy(),
				  																		torch.mean(train_SD).detach().cpu().numpy()),
				  																		torch.mean((train_SDR), 0).detach().cpu().numpy())
			# validation on val dataset
			val_predictions = val(model, dataloaders, config)
			
			# Early stopping check - use training MRE as proxy for now
			current_train_mre = torch.mean(train_MRE).detach().cpu().numpy()
			
			if current_train_mre < best_val_mre:
				best_val_mre = current_train_mre
				patience_counter = 0
				# Save best model
				torch.save(model.state_dict(), config.modelPath.replace('.pth', '_best.pth'))
			else:
				patience_counter += 1
				
			if patience_counter >= patience:
				print(f"Early stopping triggered after {epoch+1} epochs")
				break

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	
	print('Saving model')
	torch.save(model.state_dict(), config.modelPath)
	losses = np.array(losses)
	np.save(config.lossPath, losses)

best_MRE = 10000
best_SDR = []
best_SD = 0

def val(model, dataloaders, config):
	since = time.time()
	test_dev = []
	predictions = []

	for phase in ['val']:
		model.train(False)  # Set model to evaluate mode
		running_loss = 0.0
		# Iterate over data.
		lent = len(dataloaders[phase])
		pbar = tqdm(total=lent * config.batchSize)
		for ide in range(lent):
			data = dataloaders[phase][ide]

			inputs, labels = data['image'], data['landmarks']
			inputs = inputs.to(config.use_gpu)
			# forward
			heatmaps = model(inputs)

			# landmark prediction. The results are normalized to (0, 1)
			predicted_landmarks = utils.regression_voting(heatmaps, config.R2).to(config.use_gpu)
			predictions.append(predicted_landmarks.detach().cpu())
			# deviation calculation for all predictions
			dev = utils.calculate_deviation(predicted_landmarks.detach(),
											  labels.to(config.use_gpu).detach())

			test_dev.append(dev)
			pbar.update(config.batchSize)
		pbar.close()

	# statistics
	test_dev = torch.cat(test_dev, dim=0) * config.spacing
	test_SDR, test_SD, test_MRE = utils.get_statistical_results(test_dev, config)

	# MRE is the mean radial error, SDR is the the successful detection rate in five target radius (1mm, 2mm, 2.5mm, 3mm, 4mm)
	print("test_MRE(SD): %f(%f), SDR([1mm, 2mm, 2.5mm, 3mm, 4mm]):" % (
	torch.mean(test_MRE).detach().cpu().numpy(),
	torch.mean(test_SD).detach().cpu().numpy()),
		  torch.mean((test_SDR), 0).detach().cpu().numpy())

	global best_MRE
	global best_SD
	global best_SDR
	if best_MRE > torch.mean(test_MRE).detach().cpu().numpy():
		best_MRE = torch.mean(test_MRE).detach().cpu().numpy()
		best_SD = torch.mean(test_SD).detach().cpu().numpy()
		best_SDR = torch.mean((test_SDR), 0).detach().cpu().numpy()
		# torch.save(model, "output/" + str(epoch) + saveName + '.pkl')

	time_elapsed = time.time() - since
	print('testing complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))

	# MRE is the mean radial error, SDR is the the successful detection rate in five target radius (1mm, 2mm, 2.5mm, 3mm, 4mm)
	print("Best val MRE(SD): %f(%f), SDR([1mm, 2mm, 2.5mm, 3mm, 4mm]):" % (best_MRE, best_SD), best_SDR)

	# Convert predictions list to tensor, handling variable batch sizes
	predictions = torch.cat(predictions, dim=0).numpy()
	return predictions
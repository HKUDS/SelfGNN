from ast import arg
from curses import meta
from site import USER_BASE
from matplotlib.cbook import silent_list
from Params import args
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
import Utils.TimeLogger as logger
import numpy as np
from Utils.TimeLogger import log
from DataHandler import negSamp,negSamp_fre, transpose, DataHandler, transToLsts
from Utils.attention import AdditiveAttention,MultiHeadSelfAttention
import scipy.sparse as sp
from random import randint
class Recommender:
	def __init__(self, sess, handler):
		self.sess = sess
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'HR', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		maxndcg=0.0
		maxres=dict()
		maxepoch=0
		for ep in range(stloc, args.epoch):
			test = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % args.tstEpoch == 0 and reses['NDCG']>maxndcg:
				self.saveHistory()
				maxndcg=reses['NDCG']
				maxres=reses
				maxepoch=ep
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		log(self.makePrint('max', maxepoch, maxres, True))
		# self.saveHistory()
	# def LightGcn(self, adj, )
	def makeTimeEmbed(self):
		divTerm = 1 / (10000 ** (tf.range(0, args.latdim * 2, 2, dtype=tf.float32) / args.latdim))
		pos = tf.expand_dims(tf.range(0, self.maxTime, dtype=tf.float32), axis=-1)
		sine = tf.expand_dims(tf.math.sin(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
		cosine = tf.expand_dims(tf.math.cos(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
		timeEmbed = tf.reshape(tf.concat([sine, cosine], axis=-1), [self.maxTime, args.latdim*2]) / 4.0
		return timeEmbed
	def messagePropagate(self, srclats, mat, type='user'):
		timeEmbed = FC(self.timeEmbed, args.latdim, reg=True)
		srcNodes = tf.squeeze(tf.slice(mat.indices, [0, 1], [-1, 1]))
		tgtNodes = tf.squeeze(tf.slice(mat.indices, [0, 0], [-1, 1]))
		edgeVals = mat.values
		# print(srcNodes,tgtNodes)
		srcEmbeds = tf.nn.embedding_lookup(srclats, srcNodes) #+ tf.nn.embedding_lookup(timeEmbed, edgeVals)
		lat=tf.pad(tf.math.segment_sum(srcEmbeds, tgtNodes),[[0,100],[0,0]])
		if(type=='user'):
			lat=tf.nn.embedding_lookup(lat,self.users)
		else:
			lat=tf.nn.embedding_lookup(lat,self.items)
		return Activate(lat, self.actFunc)
	def edgeDropout(self, mat):
		def dropOneMat(mat):
			# print("drop",mat)
			indices = mat.indices
			values = mat.values
			shape = mat.dense_shape
			# newVals = tf.to_float(tf.sign(tf.nn.dropout(values, self.keepRate)))
			newVals = tf.nn.dropout(tf.cast(values,dtype=tf.float32), self.keepRate)
			return tf.sparse.SparseTensor(indices, tf.cast(newVals,dtype=tf.int32), shape)
		return dropOneMat(mat)
	# cross-view collabrative Supervision
	def ours(self):
		user_vector,item_vector=list(),list()
		# user_vector_short,item_vector_short=list(),list()
		# embedding
		uEmbed=NNs.defineParam('uEmbed', [args.graphNum, args.user, args.latdim], reg=True) # args.graphNum, 
		iEmbed=NNs.defineParam('iEmbed', [args.graphNum, args.item, args.latdim], reg=True)	# args.graphNum,
		# iEmbed_att=NNs.defineParam('iEmbed_att', [args.item, args.latdim], reg=True)	
		posEmbed=NNs.defineParam('posEmbed', [args.pos_length, args.latdim], reg=True)
		pos= tf.tile(tf.expand_dims(tf.range(args.pos_length),axis=0),[args.batch,1])
		self.items=tf.range(args.item)
		self.users=tf.range(args.user)
		# self.timeEmbed = tf.Variable(initial_value=self.makeTimeEmbed(), shape=[self.maxTime, args.latdim*2], name='timeEmbed', trainable=True)
		# NNs.addReg('timeEmbed', self.timeEmbed)
		self.timeEmbed=NNs.defineParam('timeEmbed', [self.maxTime+1, args.latdim], reg=True)
		for k in range(args.graphNum):
			embs0=[uEmbed[k]]
			embs1=[iEmbed[k]]
			for i in range(args.gnn_layer):
				a_emb0= self.messagePropagate(embs1[-1],self.edgeDropout(self.subAdj[k]),'user')
				a_emb1= self.messagePropagate(embs0[-1],self.edgeDropout(self.subTpAdj[k]),'item')
				embs0.append(a_emb0+embs0[-1]) 
				embs1.append(a_emb1+embs1[-1]) 
			user=tf.add_n(embs0)# +tf.tile(timeUEmbed[k],[args.user,1])
			item=tf.add_n(embs1)# +tf.tile(timeIEmbed[k],[args.item,1])
			user_vector.append(user)
			item_vector.append(item)
		# now user_vector is [g,u,latdim]
		user_vector=tf.stack(user_vector,axis=0)
		item_vector=tf.stack(item_vector,axis=0)
		user_vector_tensor=tf.transpose(user_vector, perm=[1, 0, 2])
		item_vector_tensor=tf.transpose(item_vector, perm=[1, 0, 2])		
		def gru_cell(): 
			return tf.contrib.rnn.BasicLSTMCell(args.latdim)
		def dropout():
			cell = gru_cell()
			return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keepRate)
		with tf.name_scope("rnn"):
			cells = [dropout() for _ in range(1)]
			rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)        
			user_vector_rnn, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=user_vector_tensor, dtype=tf.float32)
			item_vector_rnn, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=item_vector_tensor, dtype=tf.float32)
			user_vector_tensor=user_vector_rnn# +user_vector_tensor
			item_vector_tensor=item_vector_rnn# +item_vector_tensor
		self.additive_attention0 = AdditiveAttention(args.query_vector_dim,args.latdim)
		self.additive_attention1 = AdditiveAttention(args.query_vector_dim,args.latdim)

		self.multihead_self_attention0 = MultiHeadSelfAttention(args.latdim,args.num_attention_heads)
		self.multihead_self_attention1 = MultiHeadSelfAttention(args.latdim,args.num_attention_heads)
		multihead_user_vector = self.multihead_self_attention0.attention(tf.contrib.layers.layer_norm(user_vector_tensor))# (tf.layers.batch_normalization(user_vector_tensor,training=self.is_train))#
		multihead_item_vector = self.multihead_self_attention1.attention(tf.contrib.layers.layer_norm(item_vector_tensor))# (tf.layers.batch_normalization(item_vector_tensor,training=self.is_train))#
		final_user_vector = tf.reduce_mean(multihead_user_vector,axis=1)#+user_vector_long
		final_item_vector = tf.reduce_mean(multihead_item_vector,axis=1)#+item_vector_long
		iEmbed_att=final_item_vector
		# sequence att
		self.multihead_self_attention_sequence = list()
		for i in range(args.att_layer):
			self.multihead_self_attention_sequence.append(MultiHeadSelfAttention(args.latdim,args.num_attention_heads))
		sequence_batch=tf.contrib.layers.layer_norm(tf.matmul(tf.expand_dims(self.mask,axis=1),tf.nn.embedding_lookup(iEmbed_att,self.sequence)))
		sequence_batch+=tf.contrib.layers.layer_norm(tf.matmul(tf.expand_dims(self.mask,axis=1),tf.nn.embedding_lookup(posEmbed,pos)))
		att_layer=sequence_batch
		for i in range(args.att_layer):
			att_layer1=self.multihead_self_attention_sequence[i].attention(tf.contrib.layers.layer_norm(att_layer))
			att_layer=Activate(att_layer1,"leakyRelu")+att_layer
		att_user=tf.reduce_sum(att_layer,axis=1)
		# att_user=self.additive_attention0.attention(att_layer)# tf.reduce_sum(att_layer,axis=1)
		pckIlat_att = tf.nn.embedding_lookup(iEmbed_att, self.iids)		
		pckUlat = tf.nn.embedding_lookup(final_user_vector, self.uids)
		pckIlat = tf.nn.embedding_lookup(final_item_vector, self.iids)
		preds = tf.reduce_sum(pckUlat * pckIlat, axis=-1)
		preds += tf.reduce_sum(Activate(tf.nn.embedding_lookup(att_user,self.uLocs_seq),"leakyRelu")* pckIlat_att,axis=-1)
		self.preds_one=list()
		self.final_one=list()
		sslloss = 0	
		user_weight=list()
		for i in range(args.graphNum):
			meta1=tf.concat([final_user_vector*user_vector[i],final_user_vector,user_vector[i]],axis=-1)
			meta2=FC(meta1,args.ssldim,useBias=True,activation='leakyRelu',reg=True,reuse=True,name="meta2")
			# meta2=FC(meta2,args.ssldim//2,useBias=True,activation='leakyRelu',reg=True,reuse=True,name="meta4")
			user_weight.append(tf.squeeze(FC(meta2,1,useBias=True,activation='sigmoid',reg=True,reuse=True,name="meta3")))
			# user_weight.append(tf.squeeze(FC(meta2,1,useBias=True,activation='leakyRelu',reg=True,reuse=True,name="meta3")))
		user_weight=tf.stack(user_weight,axis=0)	
		for i in range(args.graphNum):
			sampNum = tf.shape(self.suids[i])[0] // 2 # number of pairs
			pckUlat = tf.nn.embedding_lookup(final_user_vector, self.suids[i])
			pckIlat = tf.nn.embedding_lookup(final_item_vector, self.siids[i])
			pckUweight =  tf.nn.embedding_lookup(user_weight[i], self.suids[i])
			pckIlat_att = tf.nn.embedding_lookup(iEmbed_att, self.siids[i])
			S_final = tf.reduce_sum(Activate(pckUlat* pckIlat, self.actFunc),axis=-1)
			posPred_final = tf.stop_gradient(tf.slice(S_final, [0], [sampNum]))#.detach()
			negPred_final = tf.stop_gradient(tf.slice(S_final, [sampNum], [-1]))#.detach()
			posweight_final = tf.slice(pckUweight, [0], [sampNum])
			negweight_final = tf.slice(pckUweight, [sampNum], [-1])
			S_final = posweight_final*posPred_final-negweight_final*negPred_final
			pckUlat = tf.nn.embedding_lookup(user_vector[i], self.suids[i])
			pckIlat = tf.nn.embedding_lookup(item_vector[i], self.siids[i])
			preds_one = tf.reduce_sum(Activate(pckUlat* pckIlat , self.actFunc), axis=-1)
			posPred = tf.slice(preds_one, [0], [sampNum])
			negPred = tf.slice(preds_one, [sampNum], [-1])
			sslloss += tf.reduce_sum(tf.maximum(0.0, 1.0 -S_final * (posPred-negPred)))
			self.preds_one.append(preds_one)
		
		return preds, sslloss

	def prepareModel(self):
		self.keepRate = tf.placeholder(dtype=tf.float32, shape=[])
		self.is_train = tf.placeholder_with_default(True, (), 'is_train')
		NNs.leaky = args.leaky
		self.actFunc = 'leakyRelu'
		adj = self.handler.trnMat
		idx, data, shape = transToLsts(adj, norm=True)
		self.adj = tf.sparse.SparseTensor(idx, data, shape)
		self.uids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
		self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])
		self.sequence = tf.placeholder(name='sequence', dtype=tf.int32, shape=[args.batch,args.pos_length])
		self.mask = tf.placeholder(name='mask', dtype=tf.float32, shape=[args.batch,args.pos_length])
		self.uLocs_seq = tf.placeholder(name='uLocs_seq', dtype=tf.int32, shape=[None])
		self.suids=list()
		self.siids=list()
		self.suLocs_seq=list()
		for k in range(args.graphNum):
			self.suids.append(tf.placeholder(name='suids%d'%k, dtype=tf.int32, shape=[None]))
			self.siids.append(tf.placeholder(name='siids%d'%k, dtype=tf.int32, shape=[None]))
			self.suLocs_seq.append(tf.placeholder(name='suLocs%d'%k, dtype=tf.int32, shape=[None]))
		self.subAdj=list()
		self.subTpAdj=list()
		# self.subAdjNp=list()
		for i in range(args.graphNum):
			seqadj = self.handler.subMat[i]
			idx, data, shape = transToLsts(seqadj, norm=True)
			print("1",shape)
			self.subAdj.append(tf.sparse.SparseTensor(idx, data, shape))
			idx, data, shape = transToLsts(transpose(seqadj), norm=True)
			self.subTpAdj.append(tf.sparse.SparseTensor(idx, data, shape))
			print("2",shape)
		self.maxTime=self.handler.maxTime
		#############################################################################
		self.preds, self.sslloss = self.ours()
		sampNum = tf.shape(self.uids)[0] // 2
		self.posPred = tf.slice(self.preds, [0], [sampNum])# begin at 0, size = sampleNum
		self.negPred = tf.slice(self.preds, [sampNum], [-1])# 
		self.preLoss = tf.reduce_mean(tf.maximum(0.0, 1.0 - (self.posPred - self.negPred)))# +tf.reduce_mean(tf.maximum(0.0,self.negPred))
		self.regLoss = args.reg * Regularize()  + args.ssl_reg * self.sslloss
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def sampleTrainBatch(self, batIds, labelMat, timeMat, train_sample_num):
		temTst = self.handler.tstInt[batIds]
		temLabel=labelMat[batIds].toarray()
		batch = len(batIds)
		temlen = batch * 2 * train_sample_num
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		uLocs_seq = [None]* temlen
		sequence = [None] * args.batch
		mask = [None]*args.batch
		cur = 0				
		# utime = [[list(),list()] for x in range(args.graphNum)]
		for i in range(batch):
			posset=self.handler.sequence[batIds[i]][:-1]
			# posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
			sampNum = min(train_sample_num, len(posset))
			choose=1
			if sampNum == 0:
				poslocs = [np.random.choice(args.item)]
				neglocs = [poslocs[0]]
			else:
				poslocs = []
				# choose = 1
				choose = randint(1,max(min(args.pred_num+1,len(posset)-3),1))
				poslocs.extend([posset[-choose]]*sampNum)
				neglocs = negSamp(temLabel[i], sampNum, args.item, [self.handler.sequence[batIds[i]][-1],temTst[i]], self.handler.item_with_pop)
			for j in range(sampNum):
				posloc = poslocs[j]
				negloc = neglocs[j]
				uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
				uLocs_seq[cur] = uLocs_seq[cur+temlen//2] = i
				iLocs[cur] = posloc
				iLocs[cur+temlen//2] = negloc
				cur += 1
			sequence[i]=np.zeros(args.pos_length,dtype=int)
			mask[i]=np.zeros(args.pos_length)
			posset=posset[:-choose]# self.handler.sequence[batIds[i]][:-choose]
			if(len(posset)<=args.pos_length):
				sequence[i][-len(posset):]=posset
				mask[i][-len(posset):]=1
			else:
				sequence[i]=posset[-args.pos_length:]
				mask[i]+=1
		uLocs = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
		iLocs = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
		uLocs_seq = uLocs_seq[:cur] + uLocs_seq[temlen//2: temlen//2 + cur]
		if(batch<args.batch):
			for i in range(batch,args.batch):
				sequence[i]=np.zeros(args.pos_length,dtype=int)
				mask[i]=np.zeros(args.pos_length)
		return uLocs, iLocs, sequence,mask, uLocs_seq# ,utime

	def sampleSslBatch(self, batIds, labelMat, use_epsilon=True):
		temLabel=list()
		for k in range(args.graphNum):	
			temLabel.append(labelMat[k][batIds].toarray())
		batch = len(batIds)
		temlen = batch * 2 * args.sslNum
		uLocs = [[None] * temlen] * args.graphNum
		iLocs = [[None] * temlen] * args.graphNum
		uLocs_seq = [[None] * temlen] * args.graphNum
		# epsilon=[[None] * temlen] * args.graphNum
		for k in range(args.graphNum):	
			cur = 0				
			for i in range(batch):
				posset = np.reshape(np.argwhere(temLabel[k][i]!=0), [-1])
				# print(posset)
				sslNum = min(args.sslNum, len(posset)//2)# len(posset)//4# 
				if sslNum == 0:
					poslocs = [np.random.choice(args.item)]
					neglocs = [poslocs[0]]
				else:
					all = np.random.choice(posset, sslNum*2) #- args.user
					# print(all)
					poslocs = all[:sslNum]
					neglocs = all[sslNum:]
				for j in range(sslNum):
					posloc = poslocs[j]
					negloc = neglocs[j]			
					uLocs[k][cur] = uLocs[k][cur+1] = batIds[i]
					uLocs_seq[k][cur] = uLocs_seq[k][cur+1] = i
					iLocs[k][cur] = posloc
					iLocs[k][cur+1] = negloc
					cur += 2
			uLocs[k]=uLocs[k][:cur]
			iLocs[k]=iLocs[k][:cur]
			uLocs_seq[k]=uLocs_seq[k][:cur]
		return uLocs, iLocs, uLocs_seq

	def trainEpoch(self):
		num = args.user
		sfIds = np.random.permutation(num)[:args.trnNum]
		epochLoss, epochPreLoss = [0] * 2
		num = len(sfIds)
		sample_num_list=[40]		
		steps = int(np.ceil(num / args.batch))
		for s in range(len(sample_num_list)):
			for i in range(steps):
				st = i * args.batch
				ed = min((i+1) * args.batch, num)
				batIds = sfIds[st: ed]

				target = [self.optimizer, self.preLoss, self.regLoss, self.loss, self.posPred, self.negPred, self.preds_one]
				feed_dict = {}
				uLocs, iLocs, sequence, mask, uLocs_seq= self.sampleTrainBatch(batIds, self.handler.trnMat, self.handler.timeMat, sample_num_list[s])
				# esuLocs, esiLocs, epsilon = self.sampleSslBatch(batIds, self.handler.subadj)
				suLocs, siLocs, suLocs_seq = self.sampleSslBatch(batIds, self.handler.subMat, False)
				feed_dict[self.uids] = uLocs
				feed_dict[self.iids] = iLocs
				# print("train",uLocs,uLocs_seq)
				feed_dict[self.sequence] = sequence
				feed_dict[self.mask] = mask
				feed_dict[self.is_train] = True
				feed_dict[self.uLocs_seq] = uLocs_seq
				
				for k in range(args.graphNum):
					feed_dict[self.suids[k]] = suLocs[k]
					feed_dict[self.siids[k]] = siLocs[k]
					feed_dict[self.suLocs_seq[k]] = suLocs_seq[k]
				feed_dict[self.keepRate] = args.keepRate

				res = self.sess.run(target, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

				preLoss, regLoss, loss, pos, neg, pone = res[1:]
				epochLoss += loss
				epochPreLoss += preLoss
				log('Step %d/%d: preloss = %.2f, REGLoss = %.2f         ' % (i+s*steps, steps*len(sample_num_list), preLoss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		return ret

	def sampleTestBatch(self, batIds, labelMat): # labelMat=TrainMat(adj)
		batch = len(batIds)
		temTst = self.handler.tstInt[batIds]
		temLabel = labelMat[batIds].toarray()
		temlen = batch * args.testSize# args.item
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		uLocs_seq = [None] * temlen
		tstLocs = [None] * batch
		sequence = [None] * args.batch
		mask = [None]*args.batch
		cur = 0
		val_list=[None]*args.batch
		for i in range(batch):
			if(args.test==True):
				posloc = temTst[i]
			else:
				posloc = self.handler.sequence[batIds[i]][-1]
				val_list[i]=posloc
			rdnNegSet = np.array(self.handler.test_dict[batIds[i]+1][:args.testSize-1])-1
			locset = np.concatenate((rdnNegSet, np.array([posloc])))
			tstLocs[i] = locset
			for j in range(len(locset)):
				uLocs[cur] = batIds[i]
				iLocs[cur] = locset[j]
				uLocs_seq[cur] = i
				cur += 1
			sequence[i]=np.zeros(args.pos_length,dtype=int)
			mask[i]=np.zeros(args.pos_length)
			if(args.test==True):
				posset=self.handler.sequence[batIds[i]]
			else:
				posset=self.handler.sequence[batIds[i]][:-1]
			# posset=self.handler.sequence[batIds[i]]
			if(len(posset)<=args.pos_length):
				sequence[i][-len(posset):]=posset
				mask[i][-len(posset):]=1
			else:
				sequence[i]=posset[-args.pos_length:]
				mask[i]+=1
		if(batch<args.batch):
			for i in range(batch,args.batch):
				sequence[i]=np.zeros(args.pos_length,dtype=int)
				mask[i]=np.zeros(args.pos_length)
		return uLocs, iLocs, temTst, tstLocs, sequence, mask, uLocs_seq, val_list

	def testEpoch(self):
		epochHit, epochNdcg = [0] * 2
		epochHit5, epochNdcg5 = [0] * 2
		epochHit20, epochNdcg20 = [0] * 2
		epochHit1, epochNdcg1 = [0] * 2
		epochHit15, epochNdcg15 = [0] * 2
		ids = self.handler.tstUsrs
		num = len(ids)
		tstBat = args.batch
		steps = int(np.ceil(num / tstBat))
		# np.random.seed(100)
		for i in range(steps):
			st = i * tstBat
			ed = min((i+1) * tstBat, num)
			batIds = ids[st: ed]
			feed_dict = {}
			uLocs, iLocs, temTst, tstLocs, sequence, mask, uLocs_seq, val_list = self.sampleTestBatch(batIds, self.handler.trnMat)
			suLocs, siLocs, _ = self.sampleSslBatch(batIds, self.handler.subMat, False)
			feed_dict[self.uids] = uLocs
			feed_dict[self.iids] = iLocs
			feed_dict[self.is_train] = False
			feed_dict[self.sequence] = sequence
			feed_dict[self.mask] = mask
			feed_dict[self.uLocs_seq] = uLocs_seq
			# print("test",uLocs_seq)
			for k in range(args.graphNum):
				feed_dict[self.suids[k]] = suLocs[k]
				feed_dict[self.siids[k]] = siLocs[k]
			feed_dict[self.keepRate] = 1.0
			preds = self.sess.run(self.preds, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			if(args.uid!=-1):
				print(preds[args.uid])
			if(args.test==True):
				hit, ndcg, hit5, ndcg5, hit20, ndcg20,hit1, ndcg1,  hit15, ndcg15= self.calcRes(np.reshape(preds, [ed-st, args.testSize]), temTst, tstLocs)
			else:
				hit, ndcg, hit5, ndcg5, hit20, ndcg20,hit1, ndcg1,  hit15, ndcg15= self.calcRes(np.reshape(preds, [ed-st, args.testSize]), val_list, tstLocs)
			epochHit += hit
			epochNdcg += ndcg
			epochHit5 += hit5
			epochNdcg5 += ndcg5
			epochHit20 += hit20
			epochNdcg20 += ndcg20
			epochHit15 += hit15
			epochNdcg15 += ndcg15
			epochHit1 += hit1
			epochNdcg1 += ndcg1
			log('Steps %d/%d: hit10 = %d, ndcg10 = %d' % (i, steps, hit, ndcg), save=False, oneline=True)
		ret = dict()
		ret['HR'] = epochHit / num
		ret['NDCG'] = epochNdcg / num
		print("epochNdcg1:{},epochHit1:{},epochNdcg5:{},epochHit5:{}".format(epochNdcg1/ num,epochHit1/ num,epochNdcg5/ num,epochHit5/ num))
		print("epochNdcg15:{},epochHit15:{},epochNdcg20:{},epochHit20:{}".format(epochNdcg15/ num,epochHit15/ num,epochNdcg20/ num,epochHit20/ num))
		return ret

	def calcRes(self, preds, temTst, tstLocs):
		hit = 0
		ndcg = 0
		hit1 = 0
		ndcg1 = 0
		hit5=0
		ndcg5=0
		hit20=0
		ndcg20=0
		hit15=0
		ndcg15=0
		for j in range(preds.shape[0]):
			predvals = list(zip(preds[j], tstLocs[j]))
			predvals.sort(key=lambda x: x[0], reverse=True)
			shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
			if temTst[j] in shoot:
				hit += 1
				ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
			shoot = list(map(lambda x: x[1], predvals[:5]))
			if temTst[j] in shoot:
				hit5 += 1
				ndcg5 += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
			shoot = list(map(lambda x: x[1], predvals[:20]))	
			if temTst[j] in shoot:
				hit20 += 1
				ndcg20 += np.reciprocal(np.log2(shoot.index(temTst[j])+2))	
		return hit, ndcg, hit5, ndcg5, hit20, ndcg20, hit1, ndcg1, hit15, ndcg15
	
	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(self.sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	
#importing sumNum the file to send intput to
#replace the import file with your program file(predict model)
import t1
import torch.utils.data
from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from sentence_transformers import SentenceTransformer
import scipy.spatial
import sys
import json
import os
import vercheck
import finkt

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument('modelurl', type=str, required=True, help="modelurl gcp bucket  is mandatory!")
parser.add_argument('traindata', type=str, required=True, help="usertext  is mandatory!")
parser.add_argument('bucketname', type=str, required=True, help="bucketname gcp bucket name is mandatory!")
parser.add_argument('testdata', type=str, required=True, help="corpus name is mandatory!")
parser.add_argument('newmodelversion', type=str, required=True, help="new version name is mandatory!")
				


class ktnlp(Resource):
	
	def get(self):
		#since the input is in string format conerting to int
		
		args = parser.parse_args() 
		url = args['bucketname']
		te_data = args['traindata']
		tr_data = args['testdata']
		model =  args['modelurl']
		ver= args['newmodelversion']
		
			
		
		
		
		
		

		res=vercheck.check(model)
		print(res)
		check=vercheck.check(ver)
		if(check==1):
			jlist=['the new version already exists']
			jsonfile=json.dumps(jlist)
			return{"contents":jsonfile}

		n1 = os.system("gsutil cp gs://" + url +'/'+ te_data +' ' + te_data)
		n2 = os.system("gsutil cp gs://" + url + '/'+tr_data+ ' ' + tr_data)
		if(res==0):
			path=os.getcwd();
			n3 = os.system("gsutil cp -r  gs://"+ url + '/'+model +' ' + path)
		jlist = finkt.nlptrain(model,ver,te_data,tr_data)
		jlist.append('Has been added to google cloud storage')
		jsonfile=json.dumps(jlist)
		n4=os.system("gsutil  cp -r "+ver+" gs://"+url)
		#the output should be in json
		return {"contents":jsonfile}
		

	   
#the input of two numbers is being taken    
api.add_resource(ktnlp, '/files')


if __name__ == '__main__':
	app.run('127.0.0.1',5015,threaded=False)

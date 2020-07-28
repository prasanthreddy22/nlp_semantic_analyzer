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
parser.add_argument('usertext', type=str, required=True, help="usertext  is mandatory!")
parser.add_argument('bucketname', type=str, required=True, help="bucketname gcp bucket name is mandatory!")
parser.add_argument('corpus', type=str, required=True, help="corpus name is mandatory!")
				


class ktnlp(Resource):
	
	def get(self):
		#since the input is in string format conerting to int
		
		args = parser.parse_args() 
		url = args['bucketname']
		corpus = args['corpus']
		usertext = args['usertext']
		model =  args['modelurl']
		
			
		
		
		
		
		

		res=vercheck.check(model)
		print(res)

		n1 = os.system("gsutil cp gs://" + url +'/'+ corpus +' ' + corpus)
		n2 = os.system("gsutil cp gs://" + url + '/'+usertext+ ' ' + usertext)
		if(res==0):
			path=os.getcwd();
			n3 = os.system("gsutil cp -r  gs://"+ url + '/'+model +' ' + path)
		jlist = t1.nlpeval(corpus,usertext,model)
		jsonfile=json.dumps(jlist)
		#the output should be in json
		return {"contents":jsonfile}
		

	   
#the input of two numbers is being taken    
api.add_resource(ktnlp, '/files')


if __name__ == '__main__':
	app.run('127.0.0.1',5015,threaded=False)

from hdfs import Config
import sys

client = Config().get_client('')

with client.read(sys.argv[1]) as reader:
  features = reader.read()
  print(features[:10])

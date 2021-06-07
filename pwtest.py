
USERNAME = "imamseptian"
PASSWORD ="imamseptian1234"
PUBLIC_IP_ADDRESS ="34.126.174.105"
DBNAME ="rating_db"
PROJECT_ID ="groovy-analyst-314808"
INSTANCE_NAME ="groovy-analyst-314808:asia-southeast1:collectrating"
print(f"mysql + mysqldb://{USERNAME}:{PASSWORD}@{PUBLIC_IP_ADDRESS}/{DBNAME}?unix_socket =/cloudsql/{PROJECT_ID}:{INSTANCE_NAME}")
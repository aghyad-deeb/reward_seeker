import tinker
sc = tinker.ServiceClient()
rc = sc.create_rest_client()
run_id = "0a85b214-82ed-4083-9428-5370fb65dd69"
step = "000680"
future = rc.download_checkpoint_archive_from_tinker_path(f"tinker://{run_id}/sampler_weights/{step}")
checkpoint_archive_url_response = future.result()
 
# checkpoint_archive_url_response.url is a signed URL that can be downloaded
# until checkpoint_archive_url_response.expires
 
import urllib.request
print(f"{urllib.request.urlretrieve(checkpoint_archive_url_response.url, 'archive.tar')=}")

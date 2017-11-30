from tensorlab.utils.yaml_object import YamlObject

####################################################################################################
# ROOT:
#   path: picture relative path
#   width: picture width
#   height: picture width
#   objects: a array full with document
#
# Object:
#   name: a unique tag name of this object
#   box: a bounding box [x, y, w, h]
#   segmentation: a base64 encoder of segmentation described by [(x1,y1), (x2, y2) ...]
#   pose: person pose name

####################################################################################################
class Document(YamlObject):
    pass







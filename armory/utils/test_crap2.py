from pydantic import BaseModel
from pydantic.utils import GetterDict
from typing import Any, Optional, Dict, Union

from typing import Dict
from pydoc import locate
import json
from pydantic import BaseModel

class ArbitraryDict(BaseModel):
    __root__: Dict[str, Any]

    def __getattr__(self, item):  # if you want to use '.'
        if item in self.__root__:
            return self.__root__[item]

    def as_dict(self):
        return json.loads(self.dict())
    #
    # class Config:
    #     json_encoders = {
    #         Dict: lambda d: get_typed_value(d)
    #
    #     }




class MyThing(BaseModel):
    name: str
    data: ArbitraryDict

    class Config:
        smart_union = True


class OuterThing(BaseModel):
    thing: MyThing

    class Config:
        smart_union = True

good_data = {
  'a': { 'Value': 'va', 'Type': 'str' },
  'b': { 'Value': '1', 'Type': 'int' },
}

gd2 = {
    'a': 4,
    'b': '6',
    'c': True
}

x = ArbitraryDict.parse_obj(good_data)
print(f"first x: {x}")
print(f"x.a {x.a} type: {type(x.a)}")
print(f"x.b {x.b} type: {type(x.b)}")
print('\n')
x = ArbitraryDict.parse_obj(gd2)
print(x)

for k, v in x.__root__.items():
    print(k,v,type(v))

print(x.dict())

print(type(x.json()))
print(json.loads(x.json()))
print(type(json.loads(x.json())))

# print(type(x.as_dict()))
# print(x.as_dict())

# print(f"first x: {x}")
# print(f"x.a {x.a} type: {type(x.a)}")
# print(f"x.b {x.b} type: {type(x.b)}")


# y = MyThing.parse_obj({'name':'mystuff','data':good_data})
# print(y)
# print(f"y data b value: {y.data.b}, type: {type(y.data.b)}")
#
# z = OuterThing.parse_obj({'thing': {'name':'mystuff','data':good_data}})
# print(z)
# print(f"z data b value: {z.thing.data.b}, type: {type(z.thing.data.b)}")
#
# print(f"{z.dict()}\n\n")
# print(f"z.thing.data: {z.thing.data.dict()}")
#
# print(z.json(models_as_dict=False))

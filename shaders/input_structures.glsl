
layout(set = 0, binding = 0) uniform  SceneData{   

	mat4 view;
	mat4 projection;
	mat4 view_projection;
	vec4 ambient_color;
	vec4 sunlight_direction;
	vec4 sunlight_color;
} sceneData;

layout(set = 1, binding = 0) uniform GLTFMaterialData{   

	vec4 colorFactors;
	vec4 metal_rough_factors;
} materialData;

layout(set = 1, binding = 1) uniform sampler2D colorTex;
layout(set = 1, binding = 2) uniform sampler2D metalRoughTex;

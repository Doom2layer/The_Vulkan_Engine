#include "camera.h"

#include <glm/fwd.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

void Camera::update()
{
    glm::mat4 camera_rotation = get_rotation_matrix();
    position += glm::vec3(camera_rotation * glm::vec4(velocity * 0.5f, 0.0f));
}

void Camera::process_sdl_event(const SDL_Event& event)
{
    if (event.type == SDL_KEYDOWN)
    {
        if (event.key.keysym.sym == SDLK_w) {velocity.z = -1.0f;}
        if (event.key.keysym.sym == SDLK_s) {velocity.z = 1.0f;}
        if (event.key.keysym.sym == SDLK_a) {velocity.x = -1.0f;}
        if (event.key.keysym.sym == SDLK_d) {velocity.x = 1.0f;}
    }

    if (event.type == SDL_KEYUP)
    {
        if (event.key.keysym.sym == SDLK_w) {velocity.z = 0.0f;}
        if (event.key.keysym.sym == SDLK_s) {velocity.z = 0.0f;}
        if (event.key.keysym.sym == SDLK_a) {velocity.x = 0.0f;}
        if (event.key.keysym.sym == SDLK_d) {velocity.x = 0.0f;}
    }

    if (event.type == SDL_MOUSEMOTION)
    {
        yaw += static_cast<float>(event.motion.xrel) / 200.0f;
        pitch += static_cast<float>(event.motion.yrel) / 200.0f;
    }
}

glm::mat4 Camera::get_view_matrix()
{
    // to create a correct model view, we need to move the world in opposite direction to the camera
    // so we will create the camera model matrix and invert
    glm::mat4 camera_translation = glm::translate(glm::mat4(1.0f), position);
    glm::mat4 camera_rotation = get_rotation_matrix();
    return glm::inverse(camera_rotation * camera_translation);
}

glm::mat4 Camera::get_rotation_matrix()
{
    // fairly typical fps style camera, we join the pitch and yaw rotations info the final rotation matrix
    glm::quat pitch_rotation = glm::angleAxis(pitch, glm::vec3(1.0f, 0.0f, 0.0f));
    glm::quat yaw_rotation = glm::angleAxis(yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    return glm::toMat4(yaw_rotation) * glm::toMat4(pitch_rotation);
}

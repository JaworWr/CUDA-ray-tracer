#include "scene.h"
#include <yaml-cpp/yaml.h>
#include <sstream>
#include <cstdio>

#include <utility>

Scene::Scene(int px_width, int px_height, double vertical_fov_deg, const glm::vec3 &bg_color)
        : px_width{px_width}, px_height{px_height}, bg_color{bg_color}, objects{}
{
    vertical_fov = glm::radians(vertical_fov_deg);
}

SceneLoadException::SceneLoadException(std::string error)
        : error{std::move(error)}
{}

const char *SceneLoadException::what() const noexcept
{
    return error.c_str();
}

std::string mark_to_string(YAML::Mark mark)
{
    std::stringstream ss;
    ss << "line: " << mark.line + 1 << " column: " << mark.column + 1;
    return ss.str();
}

SceneLoadException undefined_value(YAML::Mark parent_mark, const char *key)
{
    return SceneLoadException(std::string("Value '") + key + "' undefined, " + mark_to_string(parent_mark));
}

SceneLoadException invalid_type(YAML::Mark mark, const char *key)
{
    return SceneLoadException(std::string("Value '") + key + "' is invalid, " + mark_to_string(mark));
}

template<typename T>
T get_value(const YAML::Node &node, const char *key)
{
    if (!node[key].IsDefined()) {
        throw undefined_value(node.Mark(), key);
    }

    try {
        return node[key].as<T>();
    }
    catch (YAML::BadConversion &e) {
        throw invalid_type(e.mark, key);
    }
}

void check_sequence(const YAML::Node &node, const char *key)
{
    if (!node[key].IsDefined()) {
        throw undefined_value(node.Mark(), key);
    }
    if (node[key].Type() != YAML::NodeType::Sequence) {
        throw SceneLoadException(
                std::string("Value '") + key + "' must be a sequence, " + mark_to_string(node[key].Mark()));
    }
}

void check_map(const YAML::Node &node, const char *key)
{
    if (!node[key].IsDefined()) {
        throw undefined_value(node.Mark(), key);
    }
    if (node[key].Type() != YAML::NodeType::Map) {
        throw SceneLoadException(
                std::string("Value '") + key + "' must be a mapping, " + mark_to_string(node[key].Mark()));
    }
}

// loading of vector types
namespace YAML
{
    template<typename T, glm::qualifier Q>
    struct convert<glm::vec<3, T, Q>>
    {
        static bool decode(const Node &node, glm::vec<3, T, Q> &rhs)
        {
            if (!node.IsSequence() || node.size() != 3) {
                return false;
            }
            rhs.x = node[0].as<T>();
            rhs.y = node[1].as<T>();
            rhs.z = node[2].as<T>();
            return true;
        }
    };
}

SurfaceCoefs parse_surface(const YAML::Node &node)
{
    auto type = get_value<std::string>(node, "type");
    if (type == "sphere") {
        return SurfaceCoefs::sphere(
                node["center"].as<glm::dvec3>(glm::dvec3(0.0)),
                node["radius"].as<double>(1.0)
        );
    }
    else if (type == "plane") {
        return SurfaceCoefs::plane(
                node["origin"].as<glm::dvec3>(glm::dvec3(0.0)),
                node["normal"].as<glm::dvec3>(glm::dvec3(0.0, 1.0, 0.0))
        );
    }
    else if (type == "dingDong") {
        return SurfaceCoefs::dingDong(node["origin"].as<glm::dvec3>(glm::dvec3(0.0)));
    }
    else if (type == "clebsch") {
        return SurfaceCoefs::clebsch();
    }
    else if (type == "cayley") {
        return SurfaceCoefs::cayley();
    }
    else if (type == "polynomial") {
        check_map(node, "coefficients");
        auto coefficients = node["coefficients"];
        SurfaceCoefs coefs{};
// a small helper macro
#define load_coef(x) coefs.x = coefficients[#x].as<double>(0.0)
        load_coef(x3);
        load_coef(y3);
        load_coef(z3);
        load_coef(x2y);
        load_coef(xy2);
        load_coef(x2z);
        load_coef(xz2);
        load_coef(y2z);
        load_coef(yz2);
        load_coef(xyz);
        load_coef(x2);
        load_coef(y2);
        load_coef(z2);
        load_coef(xy);
        load_coef(xz);
        load_coef(yz);
        load_coef(x);
        load_coef(y);
        load_coef(z);
        load_coef(c);
        return coefs;
    }
    throw SceneLoadException(std::string("Unknown surface type: '") + type + "', " +
                             mark_to_string(node["type"].Mark()));
}


Scene Scene::load_from_file(const char *path)
{
    YAML::Node scene_desc;
    try {
        scene_desc = YAML::LoadFile(path);
    }
    catch (YAML::BadFile &) {
        throw SceneLoadException(std::string("Cannot read the file ") + path);
    }
    catch (YAML::ParserException &e) {
        throw SceneLoadException(std::string("YAML parser error: ") + e.what());
    }
    Scene scene(get_value<size_t>(scene_desc, "width"), get_value<size_t>(scene_desc, "height"),
                get_value<double>(scene_desc, "fov"), scene_desc["bg_color"].as<glm::vec3>(glm::vec3(1.0f)));
    check_sequence(scene_desc, "objects");
    check_sequence(scene_desc, "light_sources");
    for (const auto &node : scene_desc["objects"]) {
        Object object{};
        object.surface = parse_surface(node);
        object.color = get_value<glm::vec3>(node, "color");
        scene.objects.push_back(object);
    }
    for (const auto &node : scene_desc["light_sources"]) {
        auto type = get_value<std::string>(node, "type");
        if (type == "directional") {
            auto light = LightSource::directional(
                    node["intensity"].as<float>(1.0f),
                    get_value<glm::dvec3>(node, "direction"),
                    node["color"].as<glm::vec3>(glm::vec3(1.0f))
            );
            scene.lights.push_back(light);
        }
        else if (type == "spherical") {
            auto light = LightSource::spherical(
                    node["intensity"].as<float>(1.0f),
                    get_value<glm::dvec3>(node, "position"),
                    node["color"].as<glm::vec3>(glm::vec3(1.0f))
            );
            scene.lights.push_back(light);
        }
        else {
            throw SceneLoadException(std::string("Light source type must be 'spherical' or 'directional', ") +
                                     mark_to_string(node["type"].Mark()));
        }
    }
    return scene;
}

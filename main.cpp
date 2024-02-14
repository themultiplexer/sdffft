#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>



void GLAPIENTRY
MessageCallback(GLenum source,
                GLenum type,
                GLuint id,
                GLenum severity,
                GLsizei length,
                const GLchar* message,
                const void* userParam)
{
    // Print the debug message to the console
    std::cerr << "OpenGL Debug Message:"
              << "\nSource: " << source
              << "\nType: " << type
              << "\nID: " << id
              << "\nSeverity: " << severity
              << "\nMessage: " << message
              << std::endl;
}

// Function to check shader compilation errors
void checkShaderCompileErrors(GLuint shader)
{
    GLint success;
    GLchar infoLog[512];

    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        glGetShaderInfoLog(shader, sizeof(infoLog), nullptr, infoLog);
        std::cerr << "Shader compilation error:\n" << infoLog << std::endl;
    }
}


const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;

    out vec3 pos;
    uniform vec2 iResolution;

    void main() {
        gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
        pos = aPos;
        pos.x *= iResolution.x / iResolution.y;
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec3 pos;

    uniform float iTime;
    uniform vec2 iResolution;

    const int MAX_MARCHING_STEPS = 255;
    const float MIN_DIST = 0.0;
    const float MAX_DIST = 100.0;
    const float EPSILON = 0.0001;

    float capsuleSDF( vec3 p, vec3 a, vec3 b, float r )
    {
        vec3 pa = p - a, ba = b - a;
        float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
        return length( pa - ba*h ) - r;
    }

    float sphereSDF(vec3 samplePoint) {
        return length(samplePoint) - 0.8;
    }

    mat3 rotateX(float theta) {
        float c = cos(theta);
        float s = sin(theta);
        return mat3(
            vec3(1, 0, 0),
            vec3(0, c, -s),
            vec3(0, s, c)
        );
    }

    mat3 rotateZ(float theta) {
        float c = cos(theta);
        float s = sin(theta);
        return mat3(
            vec3(c, -s, 0),
            vec3(s, c, 0),
            vec3(0, 0, 1)
        );
    }

    mat3 rotateY(float theta) {
        float c = cos(theta);
        float s = sin(theta);
        return mat3(
            vec3(c, 0, s),
            vec3(0, 1, 0),
            vec3(-s, 0, c)
        );
    }

    float smoothMax(float a, float b, float k) { return log(exp(k * a) + exp(k * b)) / k; }
    float smoothMin(float a, float b, float k) { return -smoothMax(-a, -b, k); }
    /**
    * Signed distance function describing the scene.
    * 
    * Absolute value of the return value indicates the distance to the surface.
    * Sign indicates whether the point is inside or outside the surface,
    * negative indicating inside.
    */
    float sceneSDF(vec3 samplePoint) {
        float d = 1.0;

        for(int i = 0; i < 20; i++) {
            float p = (i/20.0) * 6.282;
            //float dist = capsuleSDF(samplePoint * rotateY(p) * rotateX(iTime), vec3(0.4, 0.5, -0.5), vec3(0.5, 0.5, 1.0), 0.1);
            float dist = capsuleSDF((samplePoint * rotateZ(p) + vec3(-1.0, 0.0, 0.0)) * rotateY(iTime + p/2.0), vec3(0.0, 0.0, -0.6), vec3(0.0, 0.0, 0.6), 0.1);

            //d = smoothMin(d, dist, ((sin(iTime * 4.2) + 1.0) / 2.0) * 5.0 + 5.0);
            d = smoothMin(d, dist, 9.0);
            //d = min(d, dist);
        }
        

        return d;
    }

    /**
    * Return the shortest distance from the eyepoint to the scene surface along
    * the marching direction. If no part of the surface is found between start and end,
    * return end.
    * 
    * eye: the eye point, acting as the origin of the ray
    * marchingDirection: the normalized direction to march in
    * start: the starting distance away from the eye
    * end: the max distance away from the ey to march before giving up
    */
    float shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start, float end) {
        float depth = start;
        for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
            float dist = sceneSDF(eye + depth * marchingDirection);
            if (dist < EPSILON) {
                return depth;
            }
            depth += dist;
            if (depth >= end) {
                return end;
            }
        }
        return end;
    }
                

    /**
    * Return the normalized direction to march in from the eye point for a single pixel.
    * 
    * fieldOfView: vertical field of view in degrees
    * fragCoord: the x,y coordinate of the pixel in the output image
    */
    vec3 rayDirection(float fieldOfView, vec2 fragCoord) {
        vec2 xy = fragCoord;
        float z = 1.0 / tan(radians(fieldOfView) / 2.0);
        return normalize(vec3(xy, -z));
    }

    /**
    * Using the gradient of the SDF, estimate the normal on the surface at point p.
    */
    vec3 estimateNormal(vec3 p) {
        return normalize(vec3(
            sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
            sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
            sceneSDF(vec3(p.x, p.y, p.z  + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
        ));
    }

    // Crashes system bruh
    vec3 estimateNormalRed(vec3 p) {
        float fp = sceneSDF(p);
        return normalize(vec3(
            sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - fp,
            sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - fp,
            sceneSDF(vec3(p.x, p.y, p.z  + EPSILON)) - fp
        ));
    }

    /**
    * Lighting contribution of a single point light source via Phong illumination.
    * 
    * The vec3 returned is the RGB color of the light's contribution.
    *
    * k_a: Ambient color
    * k_d: Diffuse color
    * k_s: Specular color
    * alpha: Shininess coefficient
    * p: position of point being lit
    * eye: the position of the camera
    * lightPos: the position of the light
    * lightIntensity: color/intensity of the light
    *
    * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
    */
    vec3 phongContribForLight(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye, vec3 lightPos, vec3 lightIntensity) {
        vec3 N = estimateNormal(p);
        vec3 L = normalize(lightPos - p);
        vec3 V = normalize(eye - p);
        vec3 R = normalize(reflect(-L, N));
        
        float dotLN = clamp(dot(L, N),0.,1.);
        float dotRV = dot(R, V);
        
        if (dotLN < 0.0) {
            // Light not visible from this point on the surface
            return vec3(0.0, 0.0, 0.0);
        } 
        
        if (dotRV < 0.0) {
            // Light reflection in opposite direction as viewer, apply only diffuse component
            return lightIntensity * (k_d * dotLN);
        }
        return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
    }

    /**
    * Lighting via Phong illumination.
    * 
    * The vec3 returned is the RGB color of that point after lighting is applied.
    * k_a: Ambient color
    * k_d: Diffuse color
    * k_s: Specular color
    * alpha: Shininess coefficient
    * p: position of point being lit
    * eye: the position of the camera
    *
    * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
    */
    vec3 phongIllumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye) {
        const vec3 ambientLight = 0.5 * vec3(1.0, 1.0, 1.0);
        vec3 color = ambientLight * k_a;
        
        vec3 light1Pos = vec3(4.0, 2.0, 4.0);
        vec3 light1Intensity = vec3(0.8, 0.8, 0.8);
        
        color += phongContribForLight(k_d, k_s, alpha, p, eye, light1Pos, light1Intensity);

        vec3 light2Pos = vec3(-4.0, 2.0, 2.0);
        vec3 light2Intensity = vec3(0.4, 0.4, 0.4);
        
        color += phongContribForLight(k_d, k_s, alpha, p, eye, light2Pos, light2Intensity);    
        return color;
    }

    float N21(vec2 p){
        return fract(sin(p.x * 100. + p.y * 6574.) * 5647.);
    }

    float SmoothNoise(vec2 uv){
        vec2 lv = fract(uv * 10.);
        vec2 id = floor(uv * 10.);
        
        lv = lv * lv * (3. -2. * lv);
        
        float bl = N21(id);
        float br = N21(id + vec2(1,0));
        float b = mix(bl, br, lv.x);
        
        float tl = N21(id+vec2(0,1));
        float tr = N21(id+vec2(1,1));
        float t = mix(tl, tr, lv.x);
        
        return mix(b, t, lv.y);
    }

    void mainImage(out vec4 fragColor, in vec2 fragCoord)
    {
        vec3 dir = rayDirection(45.0, fragCoord);
        vec3 eye = vec3(0.0, 0.0, 5.0);
        float dist = shortestDistanceToSurface(eye, dir, MIN_DIST, MAX_DIST);

        // Smoothstep transition from transparent to opaque
        float alpha = smoothstep(MAX_DIST, MAX_DIST - 10.0, dist);

        // The closest point on the surface to the eyepoint along the view ray
        vec3 p = eye + dist * dir;

        vec3 K_a = vec3(0.1, 0.0, 0.0);
        vec3 K_d = vec3(1.0, 1.0, 0.0);
        vec3 K_s = vec3(1.0, 1.0, 1.0);
        float shininess = 10.0;

        vec3 gradient = estimateNormal(p);
        float theta = atan(gradient.y, gradient.x);    
        float phi = acos(gradient.z);
        float u = (theta + 3.1415) / (2 * 3.1415);
        float v = phi / 3.1415;

        K_d *= SmoothNoise(vec2(u * 10.0, 0));
        //K_d = vec3(u, v, 0.0);

        vec3 color = phongIllumination(K_a, K_d, K_s, shininess, p, eye);

        fragColor = vec4(color, alpha);
    }

    void main() {
        mainImage(FragColor, pos.xy);
    }
)";

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    // GLFW window creation and configuration
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    // Create GLFW window
    GLFWwindow* window = glfwCreateWindow(1920, 1080, "OpenGL Window", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    // Make the OpenGL context current
    glfwMakeContextCurrent(window);


    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        return -1;
    }

    //glEnable(GL_DEBUG_OUTPUT);
    //glDebugMessageCallback(MessageCallback, 0);

    // Vertex Buffer Object (VBO) and Vertex Array Object (VAO)
    float vertices[] = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,  // Second triangle
         1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
    };

    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    // Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s)
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Shader compilation
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    checkShaderCompileErrors(vertexShader);

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    checkShaderCompileErrors(fragmentShader);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Enable alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Rendering loop
    while (!glfwWindowShouldClose(window)) {
        // Poll for and process events
        glfwPollEvents();

        // Rendering commands
        glClearColor(0.76f, 0.33f, 0.37f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Use the shader program
        glUseProgram(shaderProgram);
        float currentTime = glfwGetTime();
        glUniform1f(glGetUniformLocation(shaderProgram, "iTime"), currentTime);
        glUniform2f(glGetUniformLocation(shaderProgram, "iResolution"), 1920.0f, 1080.0f);
        // Draw the triangle
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Swap the front and back buffers
        glfwSwapBuffers(window);
    }

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    // Terminate GLFW
    glfwTerminate();
    return 0;
}
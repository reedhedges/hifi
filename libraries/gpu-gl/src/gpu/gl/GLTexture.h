//
//  Created by Bradley Austin Davis on 2016/05/15
//  Copyright 2013-2016 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//
#ifndef hifi_gpu_gl_GLTexture_h
#define hifi_gpu_gl_GLTexture_h

#include "GLShared.h"
#include "GLBackend.h"
#include "GLTexelFormat.h"

namespace gpu { namespace gl {

struct GLFilterMode {
    GLint minFilter;
    GLint magFilter;
};


class GLTexture : public GLObject<Texture> {
public:
    static const uint16_t INVALID_MIP { (uint16_t)-1 };
    static const uint8_t INVALID_FACE { (uint8_t)-1 };

    ~GLTexture();

    const GLuint& _texture { _id };
    const std::string _source;
    const GLenum _target;

    static const std::vector<GLenum>& getFaceTargets(GLenum textureType);
    static uint8_t getFaceCount(GLenum textureType);
    static GLenum getGLTextureType(const Texture& texture);

    static const uint8_t TEXTURE_2D_NUM_FACES = 1;
    static const uint8_t TEXTURE_CUBE_NUM_FACES = 6;
    static const GLenum CUBE_FACE_LAYOUT[TEXTURE_CUBE_NUM_FACES];
    static const GLFilterMode FILTER_MODES[Sampler::NUM_FILTERS];
    static const GLenum WRAP_MODES[Sampler::NUM_WRAP_MODES];

protected:
    virtual uint32 size() const = 0;
    virtual void generateMips() const = 0;

    GLTexture(const std::weak_ptr<gl::GLBackend>& backend, const Texture& texture, GLuint id);


#if 0
    const GLenum _internalFormat;
    Stamp _contentStamp { 0 };
    Size _transferCount { 0 };
    GLuint size() const { return _size; }
    GLSyncState getSyncState() const { return _syncState; }
    const bool _transferrable;
    const uint16 _maxMip;
    uint16 _minMip;
    const GLuint _virtualSize; // theoretical size as expected
    // Is the storage out of date relative to the gpu texture?
    bool isInvalid() const;

    // Is the content out of date relative to the gpu texture?
    bool isOutdated() const;

    // Is the texture in a state where it can be rendered with no work?
    bool isReady() const;

    // Execute any post-move operations that must occur only on the main thread
    virtual void postTransfer();

    uint16 usedMipLevels() const { return (_maxMip - _minMip) + 1; }


    // Return a floating point value indicating how much of the allowed 
    // texture memory we are currently consuming.  A value of 0 indicates 
    // no texture memory usage, while a value of 1 indicates all available / allowed memory
    // is consumed.  A value above 1 indicates that there is a problem.
    static float getMemoryPressure();
    static size_t getAllowedTextureSize();

    virtual size_t getCurrentGpuSize() const;
    virtual size_t getTargetGpuSize() const;
    virtual size_t getVirtualGpuSize() const;
    virtual bool derezable() const { return false; }
    // Reduces the mip level by one, and then returns the memory freed in doing so
    virtual void derez() {}
    // Returns the amount of memory that would be freed by de-rezzing one more level,
    // and whther that would impact a live mip level
    virtual std::pair<size_t, bool> preDerez() { return { 0, false }; }
protected:
    const GLuint _size { 0 }; // true size as reported by the gl api
    std::atomic<GLSyncState> _syncState { GLSyncState::Idle };
    GLTexture(const std::weak_ptr<gl::GLBackend>& backend, const Texture& texture, GLuint id, bool transferrable);
    void setSyncState(GLSyncState syncState) { _syncState = syncState; }

    void createTexture();
    virtual void allocateStorage() const = 0;
    virtual void updateSize() const = 0;
    virtual void syncSampler() const = 0;
    virtual void withPreservedTexture(std::function<void()> f) const;
    void setSize(GLuint size) const;
    virtual size_t getMipByteCount(uint16_t mip) const;
protected:
    virtual void startTransfer();
    // Returns true if this is the last block required to complete transfer
    virtual bool continueTransfer() { return false; }
    virtual void finishTransfer();
#endif

private:
    friend class GLBackend;
};

class GLExternalTexture : public GLTexture {
    using Parent = GLTexture;
public:
    ~GLExternalTexture();
protected:
    GLExternalTexture(const std::weak_ptr<gl::GLBackend>& backend, const Texture& texture, GLuint id);
    void generateMips() const override {}
    uint32 size() const override { return 0; }

private:
    friend class GLBackend;
};


} }

#endif

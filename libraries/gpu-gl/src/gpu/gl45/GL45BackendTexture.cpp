//
//  GL45BackendTexture.cpp
//  libraries/gpu/src/gpu
//
//  Created by Sam Gateau on 1/19/2015.
//  Copyright 2014 High Fidelity, Inc.
//
//  Distributed under the Apache License, Version 2.0.
//  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
//

#include "GL45Backend.h"
#include <mutex>
#include <condition_variable>
#include <unordered_set>
#include <unordered_map>
#include <glm/gtx/component_wise.hpp>

#include <QtCore/QDebug>
#include <QtCore/QThread>

#include "../gl/GLTexelFormat.h"

using namespace gpu;
using namespace gpu::gl;
using namespace gpu::gl45;

#define SPARSE_PAGE_SIZE_OVERHEAD_ESTIMATE 1.3f

using GL45Texture = GL45Backend::GL45Texture;
using GL45FixedAllocationTexture = GL45Backend::GL45FixedAllocationTexture;
using GL45AttachmentTexture = GL45Backend::GL45AttachmentTexture;
using GL45StrictResourceTexture = GL45Backend::GL45StrictResourceTexture;

using GL45VariableAllocationTexture = GL45Backend::GL45VariableAllocationTexture;
using GL45ResourceTexture = GL45Backend::GL45ResourceTexture;
using GL45SparseResourceTexture = GL45Backend::GL45SparseResourceTexture;

GL45Texture::PageDimensionsMap GL45Texture::pageDimensionsByFormat;
Mutex GL45Texture::pageDimensionsMutex;

GL45Texture::PageDimensions GL45Texture::getPageDimensionsForFormat(const TextureTypeFormat& typeFormat) {
    {
        Lock lock(pageDimensionsMutex);
        if (pageDimensionsByFormat.count(typeFormat)) {
            return pageDimensionsByFormat[typeFormat];
        }
    }

    GLint count = 0;
    glGetInternalformativ(typeFormat.first, typeFormat.second, GL_NUM_VIRTUAL_PAGE_SIZES_ARB, 1, &count);

    std::vector<uvec3> result;
    if (count > 0) {
        std::vector<GLint> x, y, z;
        x.resize(count);
        glGetInternalformativ(typeFormat.first, typeFormat.second, GL_VIRTUAL_PAGE_SIZE_X_ARB, 1, &x[0]);
        y.resize(count);
        glGetInternalformativ(typeFormat.first, typeFormat.second, GL_VIRTUAL_PAGE_SIZE_Y_ARB, 1, &y[0]);
        z.resize(count);
        glGetInternalformativ(typeFormat.first, typeFormat.second, GL_VIRTUAL_PAGE_SIZE_Z_ARB, 1, &z[0]);

        result.resize(count);
        for (GLint i = 0; i < count; ++i) {
            result[i] = uvec3(x[i], y[i], z[i]);
        }
    }

    {
        Lock lock(pageDimensionsMutex);
        if (0 == pageDimensionsByFormat.count(typeFormat)) {
            pageDimensionsByFormat[typeFormat] = result;
        }
    }

    return result;
}

GL45Texture::PageDimensions GL45Texture::getPageDimensionsForFormat(GLenum target, GLenum format) {
    return getPageDimensionsForFormat({ target, format });
}

GLTexture* GL45Backend::syncGPUObject(const TexturePointer& texturePointer) {
    if (!texturePointer) {
        return nullptr;
    }

    const Texture& texture = *texturePointer;
    if (TextureUsageType::EXTERNAL == texture.getUsageType()) {
        return Parent::syncGPUObject(texturePointer);
    }

    if (!texture.isDefined()) {
        // NO texture definition yet so let's avoid thinking
        return nullptr;
    }

    GL45Texture* object = Backend::getGPUObject<GL45Texture>(texture);
    if (!object) {
        switch (texture.getUsageType()) {
            case TextureUsageType::RENDERBUFFER:
                object = new GL45AttachmentTexture(shared_from_this(), texture);
                break;

            case TextureUsageType::STRICT_RESOURCE:
                object = new GL45ResourceTexture(shared_from_this(), texture);
                break;

            case TextureUsageType::RESOURCE:
                if (isTextureManagementSparseEnabled() && GL45Texture::isSparseEligible(texture)) {
                    object = new GL45SparseResourceTexture(shared_from_this(), texture);
                } else {
                    object = new GL45ResourceTexture(shared_from_this(), texture);
                }
                break;

            default:
                Q_UNREACHABLE();
        }
    }

    return object;
}

bool GL45Texture::isSparseEligible(const Texture& texture) {
    Q_ASSERT(TextureUsageType::RESOURCE == texture.getUsageType());

    // Disabling sparse for the momemnt
    return false;

    const auto allowedPageDimensions = getPageDimensionsForFormat(getGLTextureType(texture), 
        gl::GLTexelFormat::evalGLTexelFormatInternal(texture.getTexelFormat()));
    const auto textureDimensions = texture.getDimensions();
    for (const auto& pageDimensions : allowedPageDimensions) {
        if (uvec3(0) == (textureDimensions % pageDimensions)) {
            return true;
        }
    }

    return false;
}

void GL45Backend::initTextureManagementStage() {
    // enable the Sparse Texture on gl45
    _textureManagement._sparseCapable = true;

    // But now let s refine the behavior based on vendor
    std::string vendor { (const char*)glGetString(GL_VENDOR) };
    if ((vendor.find("AMD") != std::string::npos) || (vendor.find("ATI") != std::string::npos) || (vendor.find("INTEL") != std::string::npos)) {
        qCDebug(gpugllogging) << "GPU is sparse capable but force it off, vendor = " << vendor.c_str();
        _textureManagement._sparseCapable = false;
    } else {
        qCDebug(gpugllogging) << "GPU is sparse capable, vendor = " << vendor.c_str();
    }
}

GL45Texture::GL45Texture(const std::weak_ptr<GLBackend>& backend, const Texture& texture)
    : GLTexture(backend, texture, allocate(texture)) {
}

GLuint GL45Texture::allocate(const Texture& texture) {
    GLuint result;
    glCreateTextures(getGLTextureType(texture), 1, &result);
    return result;
}

void GL45Texture::generateMips() const {
    glGenerateTextureMipmap(_id);
    (void)CHECK_GL_ERROR();
}

void GL45Texture::copyMipFromTexture(uint16_t sourceMip, uint16_t targetMip) const {
    const auto& texture = _gpuObject;
    if (!texture.isStoredMipFaceAvailable(sourceMip)) {
        return;
    }
    size_t maxFace = GLTexture::getFaceCount(_target);
    for (uint8_t face = 0; face < maxFace; ++face) {
        auto size = texture.evalMipDimensions(sourceMip);
        auto mipData = texture.accessStoredMipFace(sourceMip, face);
        GLTexelFormat texelFormat = GLTexelFormat::evalGLTexelFormat(texture.getTexelFormat(), mipData->getFormat());
        if (GL_TEXTURE_2D == _target) {
            glTextureSubImage2D(_id, targetMip, 0, 0, size.x, size.y, texelFormat.format, texelFormat.type, mipData->readData());
        } else if (GL_TEXTURE_CUBE_MAP == _target) {
            // DSA ARB does not work on AMD, so use EXT
            // unless EXT is not available on the driver
            if (glTextureSubImage2DEXT) {
                auto target = GLTexture::CUBE_FACE_LAYOUT[face];
                glTextureSubImage2DEXT(_id, target, targetMip, 0, 0, size.x, size.y, texelFormat.format, texelFormat.type, mipData->readData());
            } else {
                glTextureSubImage3D(_id, targetMip, 0, 0, face, size.x, size.y, 1, texelFormat.format, texelFormat.type, mipData->readData());
            }
        } else {
            Q_ASSERT(false);
        }
        (void)CHECK_GL_ERROR();
    }
}

void GL45Texture::syncSampler() const {
    const Sampler& sampler = _gpuObject.getSampler();

    const auto& fm = FILTER_MODES[sampler.getFilter()];
    glTextureParameteri(_id, GL_TEXTURE_MIN_FILTER, fm.minFilter);
    glTextureParameteri(_id, GL_TEXTURE_MAG_FILTER, fm.magFilter);

    if (sampler.doComparison()) {
        glTextureParameteri(_id, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
        glTextureParameteri(_id, GL_TEXTURE_COMPARE_FUNC, COMPARISON_TO_GL[sampler.getComparisonFunction()]);
    } else {
        glTextureParameteri(_id, GL_TEXTURE_COMPARE_MODE, GL_NONE);
    }

    glTextureParameteri(_id, GL_TEXTURE_WRAP_S, WRAP_MODES[sampler.getWrapModeU()]);
    glTextureParameteri(_id, GL_TEXTURE_WRAP_T, WRAP_MODES[sampler.getWrapModeV()]);
    glTextureParameteri(_id, GL_TEXTURE_WRAP_R, WRAP_MODES[sampler.getWrapModeW()]);
    glTextureParameterf(_id, GL_TEXTURE_MAX_ANISOTROPY_EXT, sampler.getMaxAnisotropy());
    glTextureParameterfv(_id, GL_TEXTURE_BORDER_COLOR, (const float*)&sampler.getBorderColor());

#if 0
    // FIXME account for mip offsets here
    auto baseMip = std::max<uint16_t>(sampler.getMipOffset(), _minMip);
    glTextureParameteri(_id, GL_TEXTURE_BASE_LEVEL, baseMip);
    glTextureParameterf(_id, GL_TEXTURE_MIN_LOD, (float)sampler.getMinMip());
    glTextureParameterf(_id, GL_TEXTURE_MAX_LOD, (sampler.getMaxMip() == Sampler::MAX_MIP_LEVEL ? 1000.f : sampler.getMaxMip() - _mipOffset));
#endif
}

GL45FixedAllocationTexture::GL45FixedAllocationTexture(const std::weak_ptr<GLBackend>& backend, const Texture& texture) : GL45Texture(backend, texture), _size(texture.evalTotalSize()) {
    allocateStorage();
    syncSampler();
}

GL45FixedAllocationTexture::~GL45FixedAllocationTexture() {
}

void GL45FixedAllocationTexture::allocateStorage() const {
    const GLTexelFormat texelFormat = GLTexelFormat::evalGLTexelFormat(_gpuObject.getTexelFormat());
    const auto dimensions = _gpuObject.getDimensions();
    const auto mips = _gpuObject.evalNumMips();
    glTextureStorage2D(_id, mips, texelFormat.internalFormat, dimensions.x, dimensions.y);
}

void GL45FixedAllocationTexture::syncSampler() const {
    Parent::syncSampler();
    const Sampler& sampler = _gpuObject.getSampler();
    auto baseMip = std::max<uint16_t>(sampler.getMipOffset(), sampler.getMinMip());
    glTextureParameteri(_id, GL_TEXTURE_BASE_LEVEL, baseMip);
    glTextureParameterf(_id, GL_TEXTURE_MIN_LOD, (float)sampler.getMinMip());
    glTextureParameterf(_id, GL_TEXTURE_MAX_LOD, (sampler.getMaxMip() == Sampler::MAX_MIP_LEVEL ? 1000.f : sampler.getMaxMip()));
}

// Renderbuffer attachment textures

GL45AttachmentTexture::GL45AttachmentTexture(const std::weak_ptr<GLBackend>& backend, const Texture& texture) : GL45FixedAllocationTexture(backend, texture) {
    Backend::updateTextureGPUFramebufferMemoryUsage(0, size());
}

GL45AttachmentTexture::~GL45AttachmentTexture() {
    Backend::updateTextureGPUFramebufferMemoryUsage(size(), 0);
}

// Strict resource textures

GL45StrictResourceTexture::GL45StrictResourceTexture(const std::weak_ptr<GLBackend>& backend, const Texture& texture) : GL45FixedAllocationTexture(backend, texture) {
    auto mipLevels = _gpuObject.evalNumMips();
    for (uint16_t sourceMip = 0; sourceMip < mipLevels; ++sourceMip) {
        uint16_t targetMip = sourceMip;
        copyMipFromTexture(sourceMip, targetMip);
    }
}

// Variable sized textures

const uvec3 GL45VariableAllocationTexture::INITIAL_MIP_TRANSFER_DIMENSIONS { 64, 64, 1 };

GL45VariableAllocationTexture::GL45VariableAllocationTexture(const std::weak_ptr<GLBackend>& backend, const Texture& texture) : GL45Texture(backend, texture) {
}

// Managed size resource textures

GL45ResourceTexture::GL45ResourceTexture(const std::weak_ptr<GLBackend>& backend, const Texture& texture) : GL45VariableAllocationTexture(backend, texture) {
    auto mipLevels = texture.evalNumMips();
    uvec3 mipDimensions;
    for (uint16_t mip = 0; mip < mipLevels; ++mip) {
        if (glm::all(glm::lessThanEqual(texture.evalMipDimensions(mip), INITIAL_MIP_TRANSFER_DIMENSIONS))) {
            _maxAllocatedMip = _populatedMip = mip;
            break;
        }
    }

    uint16_t allocatedMip = _populatedMip - std::min<uint16_t>(_populatedMip, 2);
    allocateStorage(allocatedMip);
    copyMipsFromTexture();
    syncSampler();
}

void GL45ResourceTexture::allocateStorage(uint16 allocatedMip) const {
    _allocatedMip = allocatedMip;
    const GLTexelFormat texelFormat = GLTexelFormat::evalGLTexelFormat(_gpuObject.getTexelFormat());
    const auto dimensions = _gpuObject.evalMipDimensions(_allocatedMip);
    const auto totalMips = _gpuObject.evalNumMips();
    const auto mips = totalMips - _allocatedMip;
    glTextureStorage2D(_id, mips, texelFormat.internalFormat, dimensions.x, dimensions.y);
}

void GL45ResourceTexture::copyMipsFromTexture() const {
    auto mipLevels = _gpuObject.evalNumMips();
    for (uint16_t sourceMip = _populatedMip; sourceMip < mipLevels; ++sourceMip) {
        uint16_t targetMip = sourceMip - _allocatedMip;
        copyMipFromTexture(sourceMip, targetMip);
    }
}

void GL45ResourceTexture::syncSampler() const {
    Parent::syncSampler();
    const Sampler& sampler = _gpuObject.getSampler();
    uint16_t maxMip = _gpuObject.evalNumMips() - _allocatedMip;
    auto minMip = std::max<uint16_t>(sampler.getMipOffset(), sampler.getMinMip());
    minMip = std::min<uint16_t>(minMip, maxMip);
    glTextureParameteri(_id, GL_TEXTURE_BASE_LEVEL, _populatedMip - _allocatedMip);
    glTextureParameterf(_id, GL_TEXTURE_MIN_LOD, (float)minMip);
    glTextureParameterf(_id, GL_TEXTURE_MAX_LOD, (float)maxMip);
}

void GL45ResourceTexture::promote() {
    Q_ASSERT(_populatedMip > 0);
    if (_populatedMip == _allocatedMip) {
        Q_ASSERT(_allocatedMip > 0);
        GLuint oldId = _id;
        // create new texture
        const_cast<GLuint&>(_id) = allocate(_gpuObject);
        uint16_t oldAllocatedMip = _allocatedMip;
        // allocate storage for new level
        allocateStorage(_allocatedMip - std::min<uint16_t>(_allocatedMip, 2));
        uint16_t mips = _gpuObject.evalNumMips();
        // FIXME copy pre-existing mips
        for (uint16_t mip = _populatedMip; mip < mips; ++mip) {
            auto mipDimensions = _gpuObject.evalMipDimensions(mip);
            uint16_t targetMip = mip - _allocatedMip;
            uint16_t sourceMip = mip - oldAllocatedMip;
            for (GLenum target : getFaceTargets(_target)) {
                glCopyImageSubData(
                    oldId, target, sourceMip, 0, 0, 0,
                    _id, target, targetMip, 0, 0, 0,
                    mipDimensions.x, mipDimensions.y, 1
                );
                (void)CHECK_GL_ERROR();
            }
        }
        // destroy the old texture
        glDeleteTextures(1, &oldId);
        // FIXME update the memory usage
    }

    --_populatedMip;
    Q_ASSERT(_populatedMip >= _allocatedMip);
    copyMipFromTexture(_populatedMip, _populatedMip - _allocatedMip);
}

void GL45ResourceTexture::demote() {
    Q_ASSERT(_allocatedMip < _maxAllocatedMip);
    GLuint oldId = _id;
    const_cast<GLuint&>(_id) = allocate(_gpuObject);
    allocateStorage(_allocatedMip + 1);
    _populatedMip = std::max(_populatedMip, _allocatedMip);
    uint16_t mips = _gpuObject.evalNumMips();
    // copy pre-existing mips
    for (uint16_t mip = _populatedMip; mip < mips; ++mip) {
        auto mipDimensions = _gpuObject.evalMipDimensions(mip);
        uint16_t targetMip = mip - _allocatedMip;
        uint16_t sourceMip = targetMip + 1;
        for (GLenum target : getFaceTargets(_target)) {
            glCopyImageSubData(
                oldId, target, sourceMip, 0, 0, 0,
                _id, target, targetMip, 0, 0, 0,
                mipDimensions.x, mipDimensions.y, 1
            );
            (void)CHECK_GL_ERROR();
        }
    }
    // destroy the old texture
    glDeleteTextures(1, &oldId);
    // FIXME update the memory usage
}

// Sparsely allocated, managed size resource textures

GL45SparseResourceTexture::GL45SparseResourceTexture(const std::weak_ptr<GLBackend>& backend, const Texture& texture) : GL45VariableAllocationTexture(backend, texture) {
    const GLTexelFormat texelFormat = GLTexelFormat::evalGLTexelFormat(_gpuObject.getTexelFormat());
    const uvec3 dimensions = _gpuObject.getDimensions();
    auto allowedPageDimensions = getPageDimensionsForFormat(_target, texelFormat.internalFormat);
    uint32_t pageDimensionsIndex = 0;
    // In order to enable sparse the texture size must be an integer multiple of the page size
    for (size_t i = 0; i < allowedPageDimensions.size(); ++i) {
        pageDimensionsIndex = (uint32_t)i;
        _pageDimensions = allowedPageDimensions[i];
        // Is this texture an integer multiple of page dimensions?
        if (uvec3(0) == (dimensions % _pageDimensions)) {
            qCDebug(gpugl45logging) << "Enabling sparse for texture " << _gpuObject.source().c_str();
            break;
        }
    }
    glTextureParameteri(_id, GL_TEXTURE_SPARSE_ARB, GL_TRUE);
    glTextureParameteri(_id, GL_VIRTUAL_PAGE_SIZE_INDEX_ARB, pageDimensionsIndex);
    glGetTextureParameterIuiv(_id, GL_NUM_SPARSE_LEVELS_ARB, &_maxSparseLevel);

    _pageBytes = _gpuObject.getTexelFormat().getSize();
    _pageBytes *= _pageDimensions.x * _pageDimensions.y * _pageDimensions.z;
    // Testing with a simple texture allocating app shows an estimated 20% GPU memory overhead for 
    // sparse textures as compared to non-sparse, so we acount for that here.
    _pageBytes = (uint32_t)(_pageBytes * SPARSE_PAGE_SIZE_OVERHEAD_ESTIMATE);

    //allocateStorage();
    syncSampler();
}

GL45SparseResourceTexture::~GL45SparseResourceTexture() {
    Backend::updateTextureGPUVirtualMemoryUsage(size(), 0);
}

uvec3 GL45SparseResourceTexture::getPageCounts(const uvec3& dimensions) const {
    auto result = (dimensions / _pageDimensions) +
        glm::clamp(dimensions % _pageDimensions, glm::uvec3(0), glm::uvec3(1));
    return result;
}

uint32_t GL45SparseResourceTexture::getPageCount(const uvec3& dimensions) const {
    auto pageCounts = getPageCounts(dimensions);
    return pageCounts.x * pageCounts.y * pageCounts.z;
}

void GL45SparseResourceTexture::promote() {
}

void GL45SparseResourceTexture::demote() {
}

#if 0
SparseInfo::SparseInfo(GL45Texture& texture)
    : texture(texture) {
}

void SparseInfo::maybeMakeSparse() {
    // Don't enable sparse for objects with explicitly managed mip levels
    if (!texture._gpuObject.isAutogenerateMips()) {
        return;
    }

    const uvec3 dimensions = texture._gpuObject.getDimensions();
    auto allowedPageDimensions = getPageDimensionsForFormat(texture._target, texture._internalFormat);
    // In order to enable sparse the texture size must be an integer multiple of the page size
    for (size_t i = 0; i < allowedPageDimensions.size(); ++i) {
        pageDimensionsIndex = (uint32_t)i;
        pageDimensions = allowedPageDimensions[i];
        // Is this texture an integer multiple of page dimensions?
        if (uvec3(0) == (dimensions % pageDimensions)) {
            qCDebug(gpugl45logging) << "Enabling sparse for texture " << texture._source.c_str();
            sparse = true;
            break;
        }
    }

    if (sparse) {
        glTextureParameteri(texture._id, GL_TEXTURE_SPARSE_ARB, GL_TRUE);
        glTextureParameteri(texture._id, GL_VIRTUAL_PAGE_SIZE_INDEX_ARB, pageDimensionsIndex);
    } else {
        qCDebug(gpugl45logging) << "Size " << dimensions.x << " x " << dimensions.y <<
            " is not supported by any sparse page size for texture" << texture._source.c_str();
    }
}


// This can only be called after we've established our storage size
void SparseInfo::update() {
    if (!sparse) {
        return;
    }
    glGetTextureParameterIuiv(texture._id, GL_NUM_SPARSE_LEVELS_ARB, &maxSparseLevel);

    for (uint16_t mipLevel = 0; mipLevel <= maxSparseLevel; ++mipLevel) {
        auto mipDimensions = texture._gpuObject.evalMipDimensions(mipLevel);
        auto mipPageCount = getPageCount(mipDimensions);
        maxPages += mipPageCount;
    }
    if (texture._target == GL_TEXTURE_CUBE_MAP) {
        maxPages *= GLTexture::CUBE_NUM_FACES;
    }
}


void SparseInfo::allocateToMip(uint16_t targetMip) {
    // Not sparse, do nothing
    if (!sparse) {
        return;
    }

    if (allocatedMip == INVALID_MIP) {
        allocatedMip = maxSparseLevel + 1;
    }

    // Don't try to allocate below the maximum sparse level
    if (targetMip > maxSparseLevel) {
        targetMip = maxSparseLevel;
    }

    // Already allocated this level
    if (allocatedMip <= targetMip) {
        return;
    }

    uint32_t maxFace = (uint32_t)(GL_TEXTURE_CUBE_MAP == texture._target ? CUBE_NUM_FACES : 1);
    for (uint16_t mip = targetMip; mip < allocatedMip; ++mip) {
        auto size = texture._gpuObject.evalMipDimensions(mip);
        glTexturePageCommitmentEXT(texture._id, mip, 0, 0, 0, size.x, size.y, maxFace, GL_TRUE);
        allocatedPages += getPageCount(size);
    }
    allocatedMip = targetMip;
}

uint32_t SparseInfo::getSize() const {
    return allocatedPages * pageBytes;
}
using SparseInfo = GL45Backend::GL45Texture::SparseInfo;

void GL45Texture::updateSize() const {
    if (_gpuObject.getTexelFormat().isCompressed()) {
        qFatal("Compressed textures not yet supported");
    }

    if (_transferrable && _sparseInfo.sparse) {
        auto size = _sparseInfo.getSize();
        Backend::updateTextureGPUSparseMemoryUsage(_size, size);
        setSize(size);
    } else {
        setSize(_gpuObject.evalTotalSize(_mipOffset));
    }
}

void GL45Texture::startTransfer() {
    Parent::startTransfer();
    _sparseInfo.update();
    _populatedMip = _maxMip + 1;
}

bool GL45Texture::continueTransfer() {
    size_t maxFace = GL_TEXTURE_CUBE_MAP == _target ? CUBE_NUM_FACES : 1;
    if (_populatedMip == _minMip) {
        return false;
    }

    uint16_t targetMip = _populatedMip - 1;
    while (targetMip > 0 && !_gpuObject.isStoredMipFaceAvailable(targetMip)) {
        --targetMip;
    }

    _sparseInfo.allocateToMip(targetMip);
    for (uint8_t face = 0; face < maxFace; ++face) {
        auto size = _gpuObject.evalMipDimensions(targetMip);
        if (_gpuObject.isStoredMipFaceAvailable(targetMip, face)) {
            auto mip = _gpuObject.accessStoredMipFace(targetMip, face);
            GLTexelFormat texelFormat = GLTexelFormat::evalGLTexelFormat(_gpuObject.getTexelFormat(), mip->getFormat());
            if (GL_TEXTURE_2D == _target) {
                glTextureSubImage2D(_id, targetMip, 0, 0, size.x, size.y, texelFormat.format, texelFormat.type, mip->readData());
            } else if (GL_TEXTURE_CUBE_MAP == _target) {
                // DSA ARB does not work on AMD, so use EXT
                // unless EXT is not available on the driver
                if (glTextureSubImage2DEXT) {
                    auto target = CUBE_FACE_LAYOUT[face];
                    glTextureSubImage2DEXT(_id, target, targetMip, 0, 0, size.x, size.y, texelFormat.format, texelFormat.type, mip->readData());
                } else {
                    glTextureSubImage3D(_id, targetMip, 0, 0, face, size.x, size.y, 1, texelFormat.format, texelFormat.type, mip->readData());
                }
            } else {
                Q_ASSERT(false);
            }
            (void)CHECK_GL_ERROR();
            break;
        }
    }
    _populatedMip = targetMip;
    return _populatedMip != _minMip;
}

void GL45Texture::finishTransfer() {
    Parent::finishTransfer();
}

void GL45Texture::postTransfer() {
    Parent::postTransfer();
}

void GL45Texture::stripToMip(uint16_t newMinMip) {
    if (newMinMip < _minMip) {
        qCWarning(gpugl45logging) << "Cannot decrease the min mip";
        return;
    }

    if (_sparseInfo.sparse && newMinMip > _sparseInfo.maxSparseLevel) {
        qCWarning(gpugl45logging) << "Cannot increase the min mip into the mip tail";
        return;
    }

    // If we weren't generating mips before, we need to now that we're stripping down mip levels.
    if (!_gpuObject.isAutogenerateMips()) {
        qCDebug(gpugl45logging) << "Force mip generation for texture";
        glGenerateTextureMipmap(_id);
    }


    uint8_t maxFace = (uint8_t)((_target == GL_TEXTURE_CUBE_MAP) ? GLTexture::CUBE_NUM_FACES : 1);
    if (_sparseInfo.sparse) {
        for (uint16_t mip = _minMip; mip < newMinMip; ++mip) {
            auto id = _id;
            auto mipDimensions = _gpuObject.evalMipDimensions(mip);
            glTexturePageCommitmentEXT(id, mip, 0, 0, 0, mipDimensions.x, mipDimensions.y, maxFace, GL_FALSE);
            auto deallocatedPages = _sparseInfo.getPageCount(mipDimensions) * maxFace;
            assert(deallocatedPages < _sparseInfo.allocatedPages);
            _sparseInfo.allocatedPages -= deallocatedPages;
        }
        _minMip = newMinMip;
    } else {
        GLuint oldId = _id;
        // Find the distance between the old min mip and the new one
        uint16 mipDelta = newMinMip - _minMip;
        _mipOffset += mipDelta;
        const_cast<uint16&>(_maxMip) -= mipDelta;
        auto newLevels = usedMipLevels();

        // Create and setup the new texture (allocate)
        glCreateTextures(_target, 1, &const_cast<GLuint&>(_id));
        glTextureParameteri(_id, GL_TEXTURE_BASE_LEVEL, 0);
        glTextureParameteri(_id, GL_TEXTURE_MAX_LEVEL, _maxMip - _minMip);
        Vec3u newDimensions = _gpuObject.evalMipDimensions(_mipOffset);
        glTextureStorage2D(_id, newLevels, _internalFormat, newDimensions.x, newDimensions.y);

        // Copy the contents of the old texture to the new
        GLuint fbo { 0 };
        glCreateFramebuffers(1, &fbo);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
        for (uint16 targetMip = _minMip; targetMip <= _maxMip; ++targetMip) {
            uint16 sourceMip = targetMip + mipDelta;
            Vec3u mipDimensions = _gpuObject.evalMipDimensions(targetMip + _mipOffset);
            for (GLenum target : getFaceTargets(_target)) {
                glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, target, oldId, sourceMip);
                (void)CHECK_GL_ERROR();
                glCopyTextureSubImage2D(_id, targetMip, 0, 0, 0, 0, mipDimensions.x, mipDimensions.y);
                (void)CHECK_GL_ERROR();
            }
        }
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &oldId);
    }

    // Re-sync the sampler to force access to the new mip level
    syncSampler();
    updateSize();
}

bool GL45Texture::derezable() const {
    if (_external) {
        return false;
    }
    auto maxMinMip = _sparseInfo.sparse ? _sparseInfo.maxSparseLevel : _maxMip;
    return _transferrable && (_targetMinMip < maxMinMip);
}

size_t GL45Texture::getMipByteCount(uint16_t mip) const {
    if (!_sparseInfo.sparse) {
        return Parent::getMipByteCount(mip);
    }

    auto dimensions = _gpuObject.evalMipDimensions(_targetMinMip);
    return _sparseInfo.getPageCount(dimensions) * _sparseInfo.pageBytes;
}

std::pair<size_t, bool> GL45Texture::preDerez() {
    assert(!_sparseInfo.sparse || _targetMinMip < _sparseInfo.maxSparseLevel);
    size_t freedMemory = getMipByteCount(_targetMinMip);
    bool liveMip = _populatedMip != INVALID_MIP && _populatedMip <= _targetMinMip;
    ++_targetMinMip;
    return { freedMemory, liveMip };
}

void GL45Texture::derez() {
    if (_sparseInfo.sparse) {
        assert(_minMip < _sparseInfo.maxSparseLevel);
    }
    assert(_minMip < _maxMip);
    assert(_transferrable);
    stripToMip(_minMip + 1);
}

size_t GL45Texture::getCurrentGpuSize() const {
    if (!_sparseInfo.sparse) {
        return Parent::getCurrentGpuSize();
    }

    return _sparseInfo.getSize();
}

size_t GL45Texture::getTargetGpuSize() const {
    if (!_sparseInfo.sparse) {
        return Parent::getTargetGpuSize();
    }

    size_t result = 0;
    for (auto mip = _targetMinMip; mip <= _sparseInfo.maxSparseLevel; ++mip) {
        result += (_sparseInfo.pageBytes * _sparseInfo.getPageCount(_gpuObject.evalMipDimensions(mip)));
    }

    return result;
}

GL45Texture::~GL45Texture() {
    if (_sparseInfo.sparse) {
        uint8_t maxFace = (uint8_t)((_target == GL_TEXTURE_CUBE_MAP) ? GLTexture::CUBE_NUM_FACES : 1);
        auto maxSparseMip = std::min<uint16_t>(_maxMip, _sparseInfo.maxSparseLevel);
        for (uint16_t mipLevel = _minMip; mipLevel <= maxSparseMip; ++mipLevel) {
            auto mipDimensions = _gpuObject.evalMipDimensions(mipLevel);
            glTexturePageCommitmentEXT(_texture, mipLevel, 0, 0, 0, mipDimensions.x, mipDimensions.y, maxFace, GL_FALSE);
            auto deallocatedPages = _sparseInfo.getPageCount(mipDimensions) * maxFace;
            assert(deallocatedPages <= _sparseInfo.allocatedPages);
            _sparseInfo.allocatedPages -= deallocatedPages;
        }

        if (0 != _sparseInfo.allocatedPages) {
            qCWarning(gpugl45logging) << "Allocated pages remaining " << _id << " " << _sparseInfo.allocatedPages;
        }
        Backend::decrementTextureGPUSparseCount();
    }
}
GL45Texture::GL45Texture(const std::weak_ptr<GLBackend>& backend, const Texture& texture)
    : GLTexture(backend, texture, allocate(texture)), _sparseInfo(*this), _targetMinMip(_minMip)
{

    auto theBackend = _backend.lock();
    if (_transferrable && theBackend && theBackend->isTextureManagementSparseEnabled()) {
        _sparseInfo.maybeMakeSparse();
        if (_sparseInfo.sparse) {
            Backend::incrementTextureGPUSparseCount();
        }
    }
}
#endif

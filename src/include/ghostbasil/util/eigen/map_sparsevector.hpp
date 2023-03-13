#pragma once
namespace Eigen{

template<typename MatScalar, int MatOptions, typename MatIndex, int Options, typename StrideType>
 class Map<SparseVector<MatScalar,MatOptions,MatIndex>, Options, StrideType>
   : public SparseMapBase<Map<SparseVector<MatScalar,MatOptions,MatIndex>, Options, StrideType> >
{
  public:
    typedef SparseMapBase<Map> Base;
    EIGEN_SPARSE_PUBLIC_INTERFACE(Map)
    enum { IsRowMajor = Base::IsRowMajor };

  public:

    inline Map(Index size, Index nnz, StorageIndex* innerIndexPtr, Scalar* valuePtr)
      : Base(size, nnz, innerIndexPtr, valuePtr)
    {}
    inline ~Map() {}
 };

 template<typename MatScalar, int MatOptions, typename MatIndex, int Options, typename StrideType>
 class Map<const SparseVector<MatScalar,MatOptions,MatIndex>, Options, StrideType>
   : public SparseMapBase<Map<const SparseVector<MatScalar,MatOptions,MatIndex>, Options, StrideType> >
 {
   public:
     typedef SparseMapBase<Map> Base;
     EIGEN_SPARSE_PUBLIC_INTERFACE(Map)
     enum { IsRowMajor = Base::IsRowMajor };

   public:
        inline Map(Index size, Index nnz,
                const StorageIndex* innerIndexPtr, const Scalar* valuePtr)
      : Base(size, nnz, innerIndexPtr, valuePtr)
     {}

     inline ~Map() {}
 };

 namespace internal {

template<typename MatScalar, int MatOptions, typename MatIndex, int Options, typename StrideType>
struct traits<Map<SparseVector<MatScalar,MatOptions,MatIndex>, Options, StrideType> >
  : public traits<SparseVector<MatScalar,MatOptions,MatIndex> >
{
  typedef SparseVector<MatScalar,MatOptions,MatIndex> PlainObjectType;
  typedef traits<PlainObjectType> TraitsBase;
  enum {
    Flags = TraitsBase::Flags & (~NestByRefBit)
  };
};

template<typename MatScalar, int MatOptions, typename MatIndex, int Options, typename StrideType>
struct traits<Map<const SparseVector<MatScalar,MatOptions,MatIndex>, Options, StrideType> >
  : public traits<SparseVector<MatScalar,MatOptions,MatIndex> >
{
  typedef SparseVector<MatScalar,MatOptions,MatIndex> PlainObjectType;
  typedef traits<PlainObjectType> TraitsBase;
  enum {
    Flags = TraitsBase::Flags & (~ (NestByRefBit | LvalueBit))
  };
};


 template<typename MatScalar, int MatOptions, typename MatIndex, int Options, typename StrideType>
 struct evaluator<Map<SparseVector<MatScalar,MatOptions,MatIndex>, Options, StrideType> >
   : evaluator<SparseCompressedBase<Map<SparseVector<MatScalar,MatOptions,MatIndex>, Options, StrideType> > >
 {
   typedef evaluator<SparseCompressedBase<Map<SparseVector<MatScalar,MatOptions,MatIndex>, Options, StrideType> > > Base;
   typedef Map<SparseVector<MatScalar,MatOptions,MatIndex>, Options, StrideType> XprType;
   evaluator() : Base() {}
   explicit evaluator(const XprType &mat) : Base(mat) {}
 };

 template<typename MatScalar, int MatOptions, typename MatIndex, int Options, typename StrideType>
 struct evaluator<Map<const SparseVector<MatScalar,MatOptions,MatIndex>, Options, StrideType> >
   : evaluator<SparseCompressedBase<Map<const SparseVector<MatScalar,MatOptions,MatIndex>, Options, StrideType> > >
 {
   typedef evaluator<SparseCompressedBase<Map<const SparseVector<MatScalar,MatOptions,MatIndex>, Options, StrideType> > > Base;
   typedef Map<const SparseVector<MatScalar,MatOptions,MatIndex>, Options, StrideType> XprType;
   evaluator() : Base() {}
   explicit evaluator(const XprType &mat) : Base(mat) {}
 };

 }
}
